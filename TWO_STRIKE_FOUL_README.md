# 参考10場面をtest.mp4から自動で切り出し、BSO・VideoMAEで確認する

**11月13日：YOLO・OCR・VideoMAEの結果の隣に目視の正解を入力できる確認ページを追加しました。**

- [目視入力ページ用の更新ZIP](two_strike_foul_review_update.zip)
- [作成・入力の手順](two_strike_foul/MANUAL_REVIEW.md)
- [実動画5uraの確認ページ例ZIP](manual_review_example_20261113.zip)（正解欄は空欄）

**11月12日更新：野球ルールによる読取りの保留と、補正前後・適合率・再現率の表を追加しました。[実行手順](two_strike_foul/RULE_VALIDATION.md)。保存済みtest.mp4の観測だけで再確認でき、YOLO・OCRの再実行は不要です。**

- [既存プログラム用の小さい更新ZIP](two_strike_foul_rules_update.zip)（設定・重みは含まず、既存の `config.local.json` を維持）
- [参考10動画での確認表ZIP](scoreboard_rule_results_20261112.zip)（解凍して `index.html` を開く。全体の精度は未評価）

- [更新したプログラムZIP](two_strike_foul_program.zip)
- [Compact ActionFormer重みだけのZIP](compact_actionformer_weights.zip)
- [新しい実行手順](two_strike_foul/REFERENCE_EXTRACTION.md)
- [新しい検証結果](two_strike_foul/REFERENCE_VALIDATION.md)
- [実動画5uraからの抽出・分析例ZIP](reference_extraction_example.zip)

ZIPのページを開いたら、右上の **Download raw file（↓）** から保存してください。

プログラムZIPを解凍し、中の `two_strike_foul` フォルダーを `for_labelimg` の直下へ置きます。10本の参考動画が `for_labelimg` にある例：

```powershell
cd C:\Users\yutan\kenkyuu\for_labelimg\two_strike_foul
python configure_windows.py --project ".."
python run_foul_pipeline.py --config config.local.json --video "..\test.mp4" --references-dir ".." --output "..\test_reference_results" --scoreboard-scope matched
python open_results.py "..\test_reference_results"
```

以前の `config.local.json` があれば設定のやり直しは不要です。プログラムZIPには、前回の検証と同じCompact ActionFormer重みを同梱しています。元動画・YOLO重み・OCR重みは手元のものを使います。FFmpegがない場合は、表示されたコマンドで処理用環境に `imageio-ffmpeg` を入れてください。

**「Compact ActionFormer重みが見つかりません」で止まる場合：** 重みだけのZIPを現在の `two_strike_foul` フォルダーへ解凍し、`models/compact_actionformer.pth` を配置します。以前のプログラムでも、次の指定で設定できます。

```powershell
python configure_windows.py --project ".." --pitch-checkpoint ".\models\compact_actionformer.pth"
```

`設定を保存: ...config.local.json` と表示されたことを確認してから解析を実行してください。重み未検出で設定が止まると、設定ファイルも作られません。

`scenes/` に参考動画と同じ10場面、`analysis/` に投球別のBSO・投球検出・試作ファウル判別を保存します。条件とモデルが支持する投球は `analysis/foul_candidates/` に確認用MP4として保存します。`matched` は参考場面と前後だけをYOLO/OCRで調べる設定です。全体のスコアボードを記録する場合は `full` に変えます。

前の方法ではスコアボードの消失を必須にしたため、表示が残るファウルを対象にできませんでした。BSOの読み落としや、投球検出スコアとファウル判別の違いも問題でした。新モードは映像照合で投球候補の位置を求め、表示の消失を要求せずBSOを比較します。

試作ファウル判別は参考24投球で17投球が参考注釈と一致し、非ファウルの誤判定も4件ありました。出力は目視確認する候補です。test.mp4完全版はクラウドへ取り込めていないため、そこから10本を実際に書き出したとは主張しません。640×360へ再圧縮した検証用動画では10場面の照合・書き出し、実際の5uraでは1場面の照合と3投球のBSO・VideoMAE確認を行っています。

過去の消失条件版の結果は [旧検証結果ZIP](two_strike_foul_results.zip) として保持しています。
