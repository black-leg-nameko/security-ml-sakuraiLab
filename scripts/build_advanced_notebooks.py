"""Build standalone notebooks from the tested library code."""

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = " ".join((ROOT / "requirements-colab.txt").read_text().splitlines())
INSTALL = (
    "import subprocess\nimport sys\n\n"
    "subprocess.run(\n"
    f"    [sys.executable, '-m', 'pip', 'install', '-q', *{REQUIREMENTS.split()!r}],\n"
    "    check=True,\n)"
)


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def notes(chapter, section):
    path = ROOT / "notebooks" / "notes" / f"advanced_{chapter}.md"
    parts = re.split(r"(?m)^<!-- section: ([a-z_]+) -->\n", path.read_text(encoding="utf-8"))
    sections = dict(zip(parts[1::2], parts[2::2], strict=True))
    text = sections[section]
    figures = re.findall(r"\]\(\.\./figures/([^)]*)\)", text)
    for figure in figures:
        if not (ROOT / "notebooks" / "figures" / figure).is_file():
            raise FileNotFoundError(figure)
    return markdown(
        text.replace(
            "](../figures/",
            "](https://raw.githubusercontent.com/black-leg-nameko/"
            "security-ml-sakuraiLab/main/notebooks/figures/",
        )
    )


def code(text):
    source = text.strip() + "\n"
    if not source.startswith("%"):
        source = subprocess.run(
            [sys.executable, "-m", "ruff", "format", "--stdin-filename", "cell.py", "-"],
            input=source,
            text=True,
            capture_output=True,
            check=True,
            cwd=ROOT,
        ).stdout
    return nbf.v4.new_code_cell(source.rstrip("\n"))


def definitions(module):
    source = (ROOT / "src/security_ml" / f"{module}.py").read_text()
    lines = source.splitlines(keepends=True)
    nodes = ast.parse(source).body
    cells, start = [], 0
    for node in nodes:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            first = min([node.lineno, *[d.lineno for d in node.decorator_list]]) - 1
            if first > start and "".join(lines[start:first]).strip():
                cells.append(code("".join(lines[start:first])))
            cells.append(code("".join(lines[first : node.end_lineno])))
            start = node.end_lineno
    if "".join(lines[start:]).strip():
        cells.append(code("".join(lines[start:])))
    return cells


ENVIRONMENT = """
import platform
from importlib.metadata import version

import matplotlib.pyplot as plt

print(f"Python {platform.python_version()} | CPU runtime")
for package in ("numpy", "pandas", "scipy", "scikit-learn", "optuna", "lightgbm"):
    print(f"{package}: {version(package)}")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
"""


def notebook(cells):
    seen = set()
    for cell in cells:
        if cell.cell_type != "code" or cell.source.startswith("%"):
            continue
        lines = cell.source.splitlines(keepends=True)
        changes = []
        for node in ast.parse(cell.source).body:
            if not isinstance(node, ast.Import | ast.ImportFrom):
                continue
            module = node.module if isinstance(node, ast.ImportFrom) else None
            aliases = []
            for alias in node.names:
                key = (module, alias.name, alias.asname)
                if key not in seen:
                    aliases.append(alias)
                    seen.add(key)
            if len(aliases) != len(node.names):
                node.names = aliases
                replacement = ast.unparse(node) + "\n" if aliases else ""
                changes.append((node.lineno - 1, node.end_lineno, replacement))
        for first, last, replacement in reversed(changes):
            lines[first:last] = [replacement]
        cell.source = code("".join(lines)).source
    book = nbf.v4.new_notebook(cells=cells)
    book.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"provenance": []},
    }
    for i, cell in enumerate(book.cells):
        cell.id = f"cell-{i:03d}"
    return book


def chapter_one():
    return notebook(
        [
            notes("01_environment", "introduction"),
            notes("01_environment", "environment"),
            code(INSTALL),
            code(ENVIRONMENT),
            notes("01_environment", "runtime"),
            code("""
from pathlib import Path

workspace = Path.cwd() / "security-ml-lab"
workspace.mkdir(exist_ok=True)
probe = workspace / "runtime-check.txt"
probe.write_text("ready\\n", encoding="utf-8")
assert probe.read_text(encoding="utf-8") == "ready\\n"
probe.unlink()
print(f"Workspace: {workspace}")
"""),
            notes("01_environment", "workflow"),
            code("""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=500, n_features=10, n_informative=6, random_state=101)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=101,
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=101))
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(f"Synthetic holdout accuracy: {accuracy_score(y_test, prediction):.3f}")
ConfusionMatrixDisplay.from_predictions(y_test, prediction, colorbar=False)
plt.show()
"""),
            notes("01_environment", "discussion"),
        ]
    )


def chapter_two():
    cells = [
        notes("02_phishing_and_spam", "introduction"),
        code(INSTALL),
        code(ENVIRONMENT),
        notes("02_phishing_and_spam", "data"),
        *definitions("data"),
        code("""
cache_dir = Path("data")
phishing, phishing_audit = load_phishing(fetch_dataset("dataset.csv", cache_dir))
emails, email_audit = load_enron(fetch_dataset("enron1.zip", cache_dir))
display(pd.DataFrame({"phishing": vars(phishing_audit), "enron1": vars(email_audit)}).T)
display(pd.DataFrame({
    "phishing": phishing.label.value_counts(), "enron1": emails.label.value_counts(),
}).rename(index={0: "legitimate / ham", 1: "phishing / spam"}))
"""),
        notes("02_phishing_and_spam", "split"),
        *definitions("modeling"),
        code("""
config = ExperimentConfig(seed=101, trials=8, folds=3, max_false_positive_rate=0.01)
optuna.logging.set_verbosity(optuna.logging.WARNING)
phishing_split = split_indices(phishing.label, config.seed)
email_split = split_indices(emails.label, config.seed)
X_phishing = phishing.loc[:, list(FEATURE_NAMES)]
y_phishing = phishing.label
X_email = emails.text
y_email = emails.label
display(pd.DataFrame({
    "phishing": {k: len(v) for k, v in vars(phishing_split).items()},
    "enron1": {k: len(v) for k, v in vars(email_split).items()},
}).T)
"""),
        notes("02_phishing_and_spam", "classifiers"),
        code("""
models, studies = {}, {}
for kind in ("logistic", "tree"):
    models[kind], studies[kind] = tune_model(
        kind, X_phishing.iloc[phishing_split.train], y_phishing.iloc[phishing_split.train], config,
    )
    print(f"{kind}: CV F1 = {studies[kind].best_value:.4f}, params = {studies[kind].best_params}")
selected_phishing = max(("logistic", "tree"), key=lambda name: studies[name].best_value)
print(f"CV-selected phishing model: {selected_phishing}")
"""),
        notes("02_phishing_and_spam", "text"),
        code("""
models["spam"], studies["spam"] = tune_model(
    "spam", X_email.iloc[email_split.train], y_email.iloc[email_split.train], config,
)
print(f"spam: CV F1 = {studies['spam'].best_value:.4f}, params = {studies['spam'].best_params}")
"""),
        notes("02_phishing_and_spam", "threshold"),
        *definitions("evaluation"),
        code("""
tasks = {
    "logistic": (X_phishing, y_phishing, phishing_split),
    "tree": (X_phishing, y_phishing, phishing_split),
    "spam": (X_email, y_email, email_split),
}
thresholds, validation_metrics = {}, {}
for name, (X, y, split) in tasks.items():
    scores = models[name].predict_proba(X.iloc[split.validation])[:, 1]
    thresholds[name] = select_threshold(
        y.iloc[split.validation], scores, config.max_false_positive_rate,
    )
    validation_metrics[name] = binary_metrics(y.iloc[split.validation], scores, thresholds[name])
columns = ["threshold", "precision", "recall", "false_positive_rate"]
display(pd.DataFrame(validation_metrics).T[columns])
"""),
        notes("02_phishing_and_spam", "metrics"),
        code("""
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

test_metrics, test_scores = {}, {}
for name, (X, y, split) in tasks.items():
    scores = models[name].predict_proba(X.iloc[split.test])[:, 1]
    test_scores[name] = scores
    test_metrics[f"{name}/default"] = binary_metrics(y.iloc[split.test], scores)
    test_metrics[f"{name}/validation_threshold"] = binary_metrics(
        y.iloc[split.test], scores, thresholds[name],
    )
    baseline = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(split.train), 1)), y.iloc[split.train],
    )
    baseline_scores = baseline.predict_proba(np.zeros((len(split.test), 1)))[:, 1]
    test_metrics[f"{name}/majority"] = binary_metrics(y.iloc[split.test], baseline_scores)
summary = pd.DataFrame(test_metrics).T
display(summary[["accuracy", "precision", "recall", "f1", "average_precision",
                 "false_positive_rate", "fp", "fn"]].round(4))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for axis, (name, (_, y, split)) in zip(axes, tasks.items(), strict=True):
    ConfusionMatrixDisplay.from_predictions(
        y.iloc[split.test], test_scores[name] >= thresholds[name], labels=[0, 1],
        display_labels=["benign", "malicious"], colorbar=False, ax=axis,
    )
    axis.set_title(name)
fig.tight_layout()
output_dir = Path("artifacts") / "results"
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "confusion_matrices.png", bbox_inches="tight")
plt.show()

fig, axis = plt.subplots(figsize=(6, 4))
for name, (_, y, split) in tasks.items():
    PrecisionRecallDisplay.from_predictions(
        y.iloc[split.test], test_scores[name], name=name, ax=axis,
    )
fig.tight_layout()
fig.savefig(output_dir / "precision_recall.png", bbox_inches="tight")
plt.show()
"""),
        notes("02_phishing_and_spam", "bias"),
        code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for axis, (name, model) in zip(axes, models.items(), strict=True):
    importance = feature_importance(model, FEATURE_NAMES if name != "spam" else None)
    importance.to_csv(output_dir / f"{name}_importance.csv", header=True)
    importance.head(10).sort_values().plot.barh(ax=axis, title=name)
fig.tight_layout()
fig.savefig(output_dir / "feature_importance.png", bbox_inches="tight")
plt.show()

email_train = emails.iloc[email_split.train]
subject_audit = email_train.assign(
    subject_count=email_train.text.str.lower().str.count(r"\\bsubject\\b"),
).groupby("label").agg(
    messages=("text", "size"), subject_tokens=("subject_count", "sum"),
    messages_with_subject=("subject_count", lambda values: int(values.gt(0).sum())),
)
display(subject_audit.rename(index={0: "ham", 1: "spam"}))
"""),
        notes("02_phishing_and_spam", "inference"),
        code("""
import shutil

messages = pd.Series([
    "Please review the attached project agenda before tomorrow's meeting.",
    "Congratulations! Claim your free prize now. Limited offer, click to collect.",
])
scores = models["spam"].predict_proba(messages)[:, 1]
display(pd.DataFrame({
    "text": messages, "spam_score": scores, "predicted_spam": scores >= thresholds["spam"],
}))

report = save_report(
    output_dir, config, {"phishing": phishing_audit, "enron1": email_audit}, studies,
    {"phishing": phishing_split, "enron1": email_split},
    test_metrics, DATA_HASHES, UPSTREAM_REVISION,
)
summary.to_csv(output_dir / "test_metrics.csv")
subject_audit.to_csv(output_dir / "subject_audit.csv")
(output_dir / "selected_models.json").write_text(
    json.dumps({"phishing": selected_phishing, "spam": "spam", "thresholds": thresholds}, indent=2),
    encoding="utf-8",
)
archive_path = shutil.make_archive("security-ml-results", "zip", output_dir)
print(f"Results: {archive_path}")
"""),
        markdown("""
必要なら次のセルでZIPを手元に保存する。学習済みpickleやメール本文は含めず、
指標、分割インデックス、探索履歴、重要度、図、実行バージョンを記録する。
"""),
        code("""
DOWNLOAD_RESULTS = False
if DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(archive_path)
"""),
        notes("02_phishing_and_spam", "discussion"),
        markdown("""
## 出典

- [書籍サンプル・正誤情報](https://github.com/oreilly-japan/ml-security-jp)
- [UCI Phishing Websites](https://doi.org/10.24432/C51W2X) — R. Mohammad, F. Thabtah, L. McCluskey
- [Enron-Spam](https://www2.aueb.gr/users/ion/data/enron-spam/)
  — V. Metsis, I. Androutsopoulos, G. Paliouras (2006)

§2.5のLing-Spam演習は本ノートブックの対象外。
"""),
    ]
    return notebook(cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, book in (("01_environment", chapter_one()), ("02_phishing_and_spam", chapter_two())):
        path = ROOT / "notebooks" / "advanced" / f"{name}.ipynb"
        content = nbf.writes(book) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"Regenerate {path.relative_to(ROOT)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
