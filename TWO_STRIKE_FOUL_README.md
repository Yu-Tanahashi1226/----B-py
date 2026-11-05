# 参考10場面をtest.mp4から自動で切り出し、BSO・VideoMAEで確認する

- [更新したプログラムZIP](two_strike_foul_program.zip)
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

以前の `config.local.json` があれば設定のやり直しは不要です。FFmpegがない場合は、表示されたコマンドで処理用環境に `imageio-ffmpeg` を入れてください。重みの場所が異なる場合の指定方法は、実行手順にあります。元動画と既存の重みはZIPに含めていません。

`scenes/` に参考動画と同じ10場面、`analysis/` に投球別のBSO・投球検出・試作ファウル判別を保存します。条件とモデルが支持する投球は `analysis/foul_candidates/` に確認用MP4として保存します。`matched` は参考場面と前後だけをYOLO/OCRで調べる設定です。全体のスコアボードを記録する場合は `full` に変えます。

前の方法ではスコアボードの消失を必須にしたため、表示が残るファウルを対象にできませんでした。BSOの読み落としや、投球検出スコアとファウル判別の違いも問題でした。新モードは映像照合で投球候補の位置を求め、表示の消失を要求せずBSOを比較します。

試作ファウル判別は参考24投球で17投球が参考注釈と一致し、非ファウルの誤判定も4件ありました。出力は目視確認する候補です。test.mp4完全版はクラウドへ取り込めていないため、そこから10本を実際に書き出したとは主張しません。640×360へ再圧縮した検証用動画では10場面の照合・書き出し、実際の5uraでは1場面の照合と3投球のBSO・VideoMAE確認を行っています。

過去の消失条件版の結果は [旧検証結果ZIP](two_strike_foul_results.zip) として保持しています。
