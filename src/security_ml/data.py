from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

UPSTREAM_REVISION = "565def0834d063281ff0776918ad829020919752"
DATA_URL = (
    f"https://raw.githubusercontent.com/oreilly-japan/ml-security-jp/{UPSTREAM_REVISION}/ch02"
)
DATA_HASHES = {
    "dataset.csv": "98395653cdd0e5a79c73bc59907e63ce82681d40947858b73464c08981a3629c",
    "enron1.zip": "97d1e3942057e7a9192af73adc70dad1f4279a05dadde2c1606a364c245519ed",
}
FEATURE_NAMES = (
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "age_of_domain",
    "DNSRecord",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report",
)


def fetch_dataset(name: str, cache_dir: Path) -> Path:
    expected = DATA_HASHES[name]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / name
    if target.exists():
        if sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {target}. Remove the cached file and retry.")
        return target

    temporary = None
    try:
        with urlopen(f"{DATA_URL}/{name}", timeout=60) as response:
            with NamedTemporaryFile(dir=cache_dir, delete=False) as stream:
                temporary = Path(stream.name)
                digest = sha256()
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > 32 * 1024 * 1024:
                        raise ValueError("Dataset exceeds the download limit")
                    digest.update(chunk)
                    stream.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {name}")
        temporary.replace(target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class DataAudit:
    raw_rows: int
    empty_rows: int
    conflicting_rows: int
    duplicate_rows: int
    retained_rows: int


def deduplicate(frame: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, DataAudit]:
    frame = frame.copy()
    raw_rows = len(frame)
    empty = frame[keys].isna().any(axis=1)
    if keys == ["text"]:
        frame["_identity"] = frame["text"].fillna("").str.lower().str.split().str.join(" ")
        empty |= frame["_identity"].eq("")
        identity = ["_identity"]
    else:
        identity = keys
    frame = frame.loc[~empty].copy()
    conflicting = frame.groupby(identity, dropna=False)["label"].transform("nunique").gt(1)
    clean = frame.loc[~conflicting].drop_duplicates(identity)
    audit = DataAudit(
        raw_rows,
        int(empty.sum()),
        int(conflicting.sum()),
        len(frame) - int(conflicting.sum()) - len(clean),
        len(clean),
    )
    return clean.drop(columns="_identity", errors="ignore").reset_index(drop=True), audit


def load_phishing(path: Path) -> tuple[pd.DataFrame, DataAudit]:
    values = pd.read_csv(path, header=None)
    if values.shape[1] != 31 or values.empty:
        raise ValueError("Expected 30 phishing features and one label")
    if values.isna().any().any() or not values.isin([-1, 0, 1]).all().all():
        raise ValueError("Invalid phishing feature values")
    if set(values.iloc[:, -1].unique()) != {-1, 1}:
        raise ValueError("Expected UCI labels -1 (phishing) and +1 (legitimate)")
    values.columns = [*FEATURE_NAMES, "label"]
    values = values.astype(np.int8)
    values["label"] = values["label"].eq(-1).astype(np.int8)
    return deduplicate(values, list(FEATURE_NAMES))


def load_enron(path: Path) -> tuple[pd.DataFrame, DataAudit]:
    rows = []
    with ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 10_000 or sum(m.file_size for m in members) > 64 * 1024 * 1024:
            raise ValueError("Enron archive exceeds the size limit")
        for member in sorted(members, key=lambda item: item.filename):
            if member.is_dir():
                continue
            if member.filename == "enron1/Summary.txt":
                continue
            parts = PurePosixPath(member.filename).parts
            if len(parts) != 3 or parts[0] != "enron1" or parts[1] not in {"ham", "spam"}:
                raise ValueError(f"Unexpected archive member: {member.filename}")
            if member.file_size > 1024 * 1024:
                raise ValueError(f"Oversized email: {member.filename}")
            text = archive.read(member).decode("utf-8", errors="replace")
            rows.append({"text": text, "label": int(parts[1] == "spam")})
    if not rows or {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Enron archive must contain both ham and spam")
    return deduplicate(pd.DataFrame(rows), ["text"])
