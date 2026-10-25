from hashlib import sha256
from zipfile import ZipFile

import pandas as pd
import pytest

from security_ml.data import (
    DATA_HASHES,
    deduplicate,
    fetch_dataset,
    load_enron,
    load_phishing,
)


def test_duplicates_and_conflicting_labels_are_audited():
    frame = pd.DataFrame(
        {
            "text": ["Hello WORLD", " hello   world ", "ambiguous", "AMBIGUOUS", "", "unique"],
            "label": [0, 0, 0, 1, 1, 1],
        }
    )
    clean, audit = deduplicate(frame, ["text"])
    assert clean.text.tolist() == ["Hello WORLD", "unique"]
    assert vars(audit) == dict(
        raw_rows=6,
        empty_rows=1,
        conflicting_rows=2,
        duplicate_rows=1,
        retained_rows=2,
    )


def test_phishing_positive_class_is_dangerous(tmp_path):
    path = tmp_path / "phishing.csv"
    path.write_text(
        ",".join(["-1"] * 31) + "\n" + ",".join(["1"] * 31) + "\n",
        encoding="utf-8",
    )
    frame, _ = load_phishing(path)
    assert frame.label.tolist() == [1, 0]
    assert frame.shape == (2, 31)


def test_invalid_phishing_schema_fails(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="30 phishing features"):
        load_phishing(path)


def test_corrupted_cache_is_never_trusted(tmp_path):
    (tmp_path / "dataset.csv").write_bytes(b"modified")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch_dataset("dataset.csv", tmp_path)


def test_download_is_verified_before_atomic_publish(tmp_path, monkeypatch):
    from io import BytesIO

    import security_ml.data as data

    monkeypatch.setitem(DATA_HASHES, "test.csv", sha256(b"expected").hexdigest())
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch_dataset("test.csv", tmp_path)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: BytesIO(b"expected"))
    assert fetch_dataset("test.csv", tmp_path).read_bytes() == b"expected"
    assert len(list(tmp_path.iterdir())) == 1


def test_enron_archive_reads_only_expected_members(tmp_path):
    path = tmp_path / "mail.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("enron1/Summary.txt", "Dataset metadata")
        archive.writestr("enron1/ham/1.txt", "Subject: meeting")
        archive.writestr("enron1/spam/2.txt", "Subject: prize")
    frame, _ = load_enron(path)
    assert frame.label.tolist() == [0, 1]
    with ZipFile(path, "a") as archive:
        archive.writestr("../outside.txt", "unexpected")
    with pytest.raises(ValueError, match="Unexpected archive member"):
        load_enron(path)
    assert not (tmp_path.parent / "outside.txt").exists()
