"""Build standalone notebooks from the tested library code."""

import argparse
import ast
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
            markdown("""
# 第1章 — Colabの実行環境

『セキュリティエンジニアのための機械学習』第1章に対応。
Colabの **ランタイム → すべてのセルを実行** で環境確認と最小学習を実行する。
無料のCPUランタイムで十分。GPU・TPU、Google Drive、APIキーは不要。
インストールは新しいランタイムの最初に実行する。
"""),
            code(INSTALL),
            code(ENVIRONMENT),
            markdown("""
## 作業領域とランタイム

Colabのファイルはセッション終了で失われる。第2章では結果をZIPにまとめて保存できる。
依存パッケージはノートブック内に固定し、実行したバージョンも結果に記録する。
ランタイムの制限・仕様は [Colab FAQ](https://research.google.com/colaboratory/faq.html) を参照。
"""),
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
            markdown("""
## 学習 → 未知データによる評価

動作確認用の合成データを使う。セキュリティ上の検知性能を表すデータではない。
標準化は訓練データだけから学習し、テストデータへ適用する。
"""),
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
            markdown("""
第2章では実データの読み込み、重複・ラベルの監査、ハイパーパラメータ探索、
閾値選択、テスト評価を順に進める。第2章のノートブックも単独で実行できる。
"""),
        ]
    )


def chapter_two():
    cells = [
        markdown("""
# 第2章 — フィッシングサイトと迷惑メールの検出

『セキュリティエンジニアのための機械学習』§2.2–2.3の独立した再実装。
**ランタイム → すべてのセルを実行**。CPUで数分が目安（環境により変動）。
第1章の事前実行、GitHub認証、手動ファイル配置は不要。

| 対象 | 入力 | モデル | 陽性（1） |
|---|---|---|---|
| フィッシング | UCIの30個の既存特徴量 | ロジスティック回帰・決定木 | 元ラベル -1 |
| 迷惑メール | Enron1の英語メール | TF-IDF + LightGBM | spamフォルダ |

URLを入力して特徴量を収集する実装ではない。実サイトへの接続やメール送信は行わない。
データの配布元・固定コミット・SHA-256をコードに記載する。
"""),
        code(INSTALL),
        code(ENVIRONMENT),
        markdown("""
## 1. データ取得と監査

出版社の公開リポジトリからデータだけを取得する。SHA-256不一致は停止し、ZIPは展開せず読む。
空データを除き、同じ入力でラベルが矛盾する行は全件除外、残りの重複は1件にまとめる。
メールの重複判定は大文字小文字と空白の違いを無視する。本文自体は改変しない。

**UCIの -1 はフィッシング、+1 は正常**。陽性が危険側を示すよう0/1に変換する。
同じ特徴量の行を分割間にまたがせないため、書籍の数値とは直接比較できない。
"""),
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
        markdown("""
## 2. 訓練・検証・テストの分離

60%を訓練、20%を閾値選択、20%を最終テストに使い、クラス比率を維持する。
Optunaは訓練部分の層化3分割CVでF1を最大化する。標準化とTF-IDFの語彙・IDFも各fold内で学習する。
探索の並列化は行わず、乱数と試行回数を固定する。TF-IDFは疎行列のままLightGBMへ渡す。
"""),
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
        markdown("## 3. ロジスティック回帰・決定木の探索と学習"),
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
        markdown("## 4. TF-IDF + LightGBMの探索と学習"),
        code("""
models["spam"], studies["spam"] = tune_model(
    "spam", X_email.iloc[email_split.train], y_email.iloc[email_split.train], config,
)
print(f"spam: CV F1 = {studies['spam'].best_value:.4f}, params = {studies['spam'].best_params}")
"""),
        markdown("""
## 5. 検証データで閾値を決める

検証データ上の誤検知率が1%以下になる点のうち、再現率が最大の閾値を採用する。
同点なら高い閾値を使う。陽性判定は `score >= threshold`。
これは検証標本上の制約であり、テストや実運用の誤検知率を保証しない。
スコアは較正した確率ではない。閾値を決めた後はモデルを再学習しない。
"""),
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
        markdown("""
## 6. 最終テスト

通常の閾値0.5と検証で決めた閾値を併記する。平均適合率（AP）はスコアの順位全体を評価する。
比較用の多数派クラス分類器も訓練データだけで学習する。
テスト結果を使ってモデル・探索範囲・閾値を選び直さない。
"""),
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
        markdown("""
## 7. 特徴量とデータ由来の偏り

ロジスティック回帰は標準化後の係数の絶対値、決定木は不純度の減少、LightGBMはgainを表示する。
重要度は因果関係を示さず、モデルをまたいで値の大きさを比較できない。
書籍が注目する `subject` の出現数も訓練データだけで確認する。
この単語の偏りだけで検知回避が可能とは結論できない。
"""),
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
        markdown("""
## 8. 推論と結果の保存

以下は手入力した英語テキストの推論例。判定の正しさを保証するものではない。
学習データは古く、送信者・時系列・キャンペーンの独立性も確認できない。
重複排除後も近似重複やデータセット特有の偏りは残る。現代の実メールや日本語メールへの性能は未検証。
フィッシングの30特徴量には現在取得しにくいものも含まれるため、実運用には別途設計が必要。
"""),
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
        path = ROOT / "notebooks" / f"{name}.ipynb"
        content = nbf.writes(book) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"Regenerate {path.relative_to(ROOT)}")
        else:
            path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
