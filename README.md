# セキュリティエンジニアのための機械学習 — Colab実習

第1章の環境構築と、第2章「フィッシングサイトと迷惑メールの検出」の実装。
ノートブックは単独で動作する。Colabで開き、**ランタイム → すべてのセルを実行**。
無料のCPUランタイムでよく、GPU、Driveマウント、APIキー、リポジトリのcloneは不要。

| ノートブック | 内容 | Colab |
|---|---|---|
| [01_environment.ipynb](notebooks/01_environment.ipynb) | パッケージ導入、実行環境、最小学習 | [開く](https://colab.research.google.com/github/black-leg-nameko/security-ml-sakuraiLab/blob/main/notebooks/01_environment.ipynb) |
| [02_phishing_and_spam.ipynb](notebooks/02_phishing_and_spam.ipynb) | ロジスティック回帰、決定木、TF-IDF＋LightGBM | [開く](https://colab.research.google.com/github/black-leg-nameko/security-ml-sakuraiLab/blob/main/notebooks/02_phishing_and_spam.ipynb) |

第2章はCPUで数分が目安。初回はパッケージ・データをダウンロードするためネット接続が必要。
新しいランタイムで最初のセルから実行する。既存セッションで別バージョンを読み込んでいた場合は、
ランタイムを再起動してから実行する。

## 実装範囲

| 書籍 | 実装 |
|---|---|
| 第1章 | Colab環境構築、依存バージョンの表示、作業領域、訓練と評価の動作確認 |
| §2.2.1 | ロジスティック回帰＋Optuna |
| §2.2.2 | 決定木＋Optuna |
| §2.3.2 | Enron1、TF-IDF＋LightGBM＋Optuna、特徴量重要度、subjectの偏りの確認 |
| §2.5 | Ling-Spamの練習問題は対象外 |

提供PDFの内容を踏まえた独立した再実装。書籍本文・PDFやデータセットはリポジトリに含めない。
実サイトの巡回、URLからの特徴量抽出、実メールの送受信は行わない。

## 評価の設計

- **陽性は危険側**。UCIの元ラベル `-1` を `phishing=1` に変換し、Enronのspamも `1` とする。
- 同じ特徴量ベクトル、または空白と大小文字を正規化した同じメールを分割間にまたがせない。
  ラベルが矛盾する同一入力は全件除外し、除外数を記録する。
- 層化分割で訓練60%、閾値選択20%、最終テスト20%。Optunaは訓練内の層化3分割CVのみでF1を最適化する。
- 標準化、TF-IDFの語彙、IDFを各foldの訓練部分だけで学習する。TF-IDFは疎行列のまま扱う。
- 試行数・分割・モデルの乱数を固定する。時間打ち切りや探索の並列実行による試行数の変動を避ける。
- 閾値は検証上の誤検知率1%以下で再現率が最大になる点を選び、テスト前に固定する。
  フィッシングの採用モデルはCV F1で決める。テストを見て選び直さない。
- Accuracy、Precision、Recall、F1、AP、ROC AUC、誤検知率、FP/FNを保存し、多数派分類器と比較する。
  閾値0.5と検証で選んだ閾値の両方を報告する。

検証上の誤検知率制約は未知データへの保証ではない。予測スコアは較正していない。
重複除去と危険側へのラベル変換、分割方法が異なるため、書籍掲載の精度との直接比較はできない。
探索する指標は通常閾値のCV F1であり、低誤検知率での再現率に最適なモデルとは限らない。

## データの出所

データは [oreilly-japan/ml-security-jp](https://github.com/oreilly-japan/ml-security-jp) の
コミット `565def0834d063281ff0776918ad829020919752` に固定。
取得後は [data.py](src/security_ml/data.py) のSHA-256と照合する。不一致では停止する。
ZIPをディスクへ展開せず、ファイル数・展開後サイズ・個別サイズとパス構造を検査して読む。

- **Phishing Websites** — R. Mohammad, F. Thabtah, L. McCluskey、
  [UCI DOI: 10.24432/C51W2X](https://doi.org/10.24432/C51W2X)。元CSVは11,055行・30特徴量。
  UCI配布の `Training Dataset.arff` と出版社CSVの全値が一致することを確認済み。
  列名はUCIの表記を維持している。
- **Enron-Spam / enron1** — V. Metsis, I. Androutsopoulos, G. Paliouras (2006)、
  [配布元](https://www2.aueb.gr/users/ion/data/enron-spam/)。ham 3,672通、spam 1,500通。

古いデータセットを用いた学習用ベンチマークであり、現代のフィッシングや実メールへの性能は未検証。
時系列・送信者・キャンペーン単位の独立性は確認できず、近似重複も残りうる。
英語のEnronデータに対する実装で、日本語メールには対応していない。
重要度や `subject` の偏りは因果関係や検知回避の成立を示すものではない。
データの再利用は各配布元の条件を確認すること。

## 結果の持ち出し

第2章の実行後に `security-ml-results.zip` を生成する。
最後のセルで `DOWNLOAD_RESULTS = True` にすればColabからダウンロードできる。

ZIPには `report.json`、指標CSV、Optunaの試行履歴、特徴量重要度、subject集計、図を含む。
JSONにはデータのハッシュ、除去件数、分割インデックス、乱数、探索結果、実行バージョンを保存する。
メール本文と学習済みpickleは含めない。Colabの作業ファイルはセッション終了で失われる。

## ローカル実行・変更

Python 3.12で検証する。LinuxではLightGBMが `libgomp.so.1` を必要とする（Colabには導入済み）。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-colab.txt -e '.[dev]'
pytest -q
ruff check .
ruff format --check .
python scripts/build_notebooks.py --check
python scripts/execute_notebooks.py
```

ライブラリを変更したら `python scripts/build_notebooks.py` でノートブックを再生成する。
生成時にライブラリのコードを通常のコードセルとして埋め込むため、Colabではこのパッケージの
インストールやGitHubからのコード取得が不要。生成物を直接編集しない。
`ExperimentConfig` の `trials`、`folds`、`max_false_positive_rate` で計算量と閾値方針を変更できる。

CIは単体テスト、書式、生成物の一致、2つのノートブックの全セル実行を検証する。
実行済みノートブックとレポートはGitHub Actionsのartifactに保存する。
実測値と検証条件は [VALIDATION.md](VALIDATION.md) を参照。
