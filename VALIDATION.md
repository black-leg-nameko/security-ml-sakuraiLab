# 本書準拠版の検証

本編は `notebooks/01_environment.ipynb` と `notebooks/02_phishing_and_spam.ipynb`。
出版社の配布コード（コミット `565def0834d063281ff0776918ad829020919752`）と、提供PDFを照合した。
配布コードのコードセルは `notebooks/book/02_code.json` に保存している。

第2章のコードセルは25個。18個は配布コードと同一で、7個だけを以下の理由で修正している。
セル番号は配布コードの0始まりの番号である。

| セル | 差分 | 理由 |
|---|---|---|
| 0・10 | `wget -O` | 再実行時も同名ファイルに取得する |
| 1 | Optuna、optuna-integration、LightGBMの導入 | LightGBM連携のパッケージ分離への対応。インストールは本書同様`!pip` |
| 8 | `max_features`の`auto`を除く | 旧`auto`と同じ意味の`sqrt`へ候補を統合 |
| 11 | `unzip -oq` | 再実行時の上書き確認を避ける |
| 17 | `get_feature_names_out()`、`lowercase=False`を削除 | 削除されたAPIへの対応と、PDFの設定への一致 |
| 21 | `callbacks=[lgb.log_evaluation(0)]` | 削除された`verbose_eval`への対応 |

80:20分割、5分割CV、正解率による探索、各60秒のOptuna探索、`Objective`、`Objective_DTC`、
`LightGBMTunerCV`、密なDataFrameへの変換、`np.rint`、本文の`subject`集計は本書の流れを維持した。
Pipeline、重複除去、ラベル変換、独立した検証部分、閾値最適化は本編には加えていない。

サイトの元ラベルは `-1=フィッシング`、`+1=正常`。
本書の `precision_score` と `recall_score` は既定の `pos_label=1` を使うため、正常側の指標になる。
また、LightGBMTunerCVの `1 - best_score` は、既定のbinary loglossから算出する値で、正解率ではない。
どちらもコードを独自に置き換えず、対応する説明セルに注記した。

生成物は `python scripts/build_notebooks.py --check` で確認できる。
本書の空白、importの順序、重複するimportを保つため、本編2つのノートブックはRuffの整形対象から外している。
ライブラリと生成スクリプトのlint・formatは引き続き検査する。

発展編は `notebooks/advanced/` に分離した。発展編の以前の実測結果は同ディレクトリの
`VALIDATION.md` を参照。本編と評価方法が異なるため、指標を直接比較しない。
