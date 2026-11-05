# 2ストライク後のファウル候補に絞る統合プログラム

- [プログラムZIP](two_strike_foul_program.zip)（約1.6MB）
- [10本の検証結果ZIP](two_strike_foul_results.zip)（約22.3MB）
- [Windowsでの実行手順](two_strike_foul/README.md)
- [実行結果と、条件だけでは拾えなかった理由](two_strike_foul/VALIDATION.md)
- [追加アノテーションを検討する画像](two_strike_foul/validation/manual_cases/README.md)

ZIPのページを開いたら、右上の **Download raw file（↓）** から保存してください。

プログラムZIPを解凍し、中の `two_strike_foul` フォルダーを `C:\Users\yutan\kenkyuu\for_labelimg` の直下に置く例：

```powershell
cd C:\Users\yutan\kenkyuu\for_labelimg\two_strike_foul
python configure_windows.py --project "C:\Users\yutan\kenkyuu\for_labelimg"
python run_foul_pipeline.py --config config.local.json --preflight
python run_foul_pipeline.py --config config.local.json --video "..\test.mp4" --output "..\results_test_foul_only"
python open_results.py "..\results_test_foul_only"
```

重みの保存場所を自動で見つけられない場合の指定方法は、実行手順にあります。重みと元動画はZIPには含めていません。

既存の10本を使った検証では、条件に一致した2場面・計16.5秒だけにVideoMAEを適用しました。対象外への入力は0件です。表示が消えない場面、動画の終端で再表示後を比較できない場面もあるため、条件を守ったまま24投球すべてを選べたわけではありません。test.mp4の完全版は、このクラウド検証では実行していません。

結果ZIPは解凍後、`two_strike_foul_results/index.html` をブラウザーで開くと、10本の結果へ移動できます。元動画を選択すれば、各場面を再生して確認できます。各動画フォルダーの `annotation_review` に確認用画像と表を入れています。
