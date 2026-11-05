# 実行した検証（2026年11月5日）

既定の「消失前が2ストライク、再表示後もBSO不変」の条件で、保存済みの10本を最後まで処理しました。YOLO・OCRは全動画の0.5秒間隔の観測に適用し、VideoMAEは条件を満たした範囲だけに適用しました。

| 入力動画 | 長さ（秒） | 消失・再表示を確認した場面 | VideoMAE対象 | ファウル推定 |
|---|---:|---:|---:|---:|
| [1urafaul1.mp4](validation/results/1urafaul1/summary.json) | 20.97 | 1 | 1 | 1 |
| [2omotefaul2.mp4](validation/results/2omotefaul2/summary.json) | 64.20 | 2 | 1 | 1 |
| [2urafaul3.mp4](validation/results/2urafaul3/summary.json) | 25.57 | 0 | 0 | 0 |
| [4omotefaul4.mp4](validation/results/4omotefaul4/summary.json) | 57.70 | 1 | 0 | 0 |
| [4urafaul5.mp4](validation/results/4urafaul5/summary.json) | 21.53 | 0 | 0 | 0 |
| [5omotefaul6.mp4](validation/results/5omotefaul6/summary.json) | 27.97 | 0 | 0 | 0 |
| [5omotefaul7.mp4](validation/results/5omotefaul7/summary.json) | 54.17 | 0 | 0 | 0 |
| [5urafaul8.mp4](validation/results/5urafaul8/summary.json) | 57.37 | 0 | 0 | 0 |
| [7urafaul9.mp4](validation/results/7urafaul9/summary.json) | 25.27 | 0 | 0 | 0 |
| [8urafaul10.mp4](validation/results/8urafaul10/summary.json) | 23.80 | 0 | 0 | 0 |

合計378.53秒、762観測。VideoMAE対象は2場面・計16.5秒、16フレーム入力は59個でした。対象外へ渡した入力窓は **0個** です。

## 投球モデルを実際に実行した2場面

| 動画 | VideoMAEに渡した範囲 | 最も高い投球候補の区間 | 投球スコア |
|---|---|---|---:|
| 1urafaul1.mp4 | 0.00〜7.50秒 | 1.820〜4.116秒 | 0.941082 |
| 2omotefaul2.mp4 | 0.00〜9.00秒 | 3.713〜5.886秒 | 0.904691 |

推定は「表示条件に一致し、その付近に投球らしい動作を検出した」という意味です。スコアをファウルの確率とは扱いません。確認済みのファウル数、全体の精度・再現率・AP・平均tIoUは今回算出していません。

## 24投球すべてを対象にできなかった理由

表示が消えない場面がありました。たとえば2omotefaul2の2球目付近は0.5秒間隔で表示の検出が続き、35.5秒の画像にも表示が残っています。2urafaul3の1球目後6.0秒にも表示が残っています。**表示消失を必須にする条件だけでは、この種類のファウルを選べません。**

さらに2omotefaul2の最後の再表示は63.0〜64.0秒、4omotefaul4の再表示は52.0〜52.5秒でした。既定で再表示から1秒待ち、その後3回連続の安定値を求めるため、これらは比較に必要な観測が不足しました。後者のようにすぐ表示が消える場合や、前者のように動画が終わる場合もあり、BSOの学習不足とは区別すべきです。

表示条件を満たさない場面まで広げて、24投球という既存の数に合わせる処理は行っていません。比較用に次の投球を追加するオプションは、既定のファウル条件とは分けてあります。今回の10本のモデル実行は既定モードです。比較オプションの分岐・範囲は自動テストで確認しました。

## 追加注釈を検討する具体例

[画像と目視確認の記録](validation/manual_cases/README.md) にまとめています。

- **BSO検出の失敗**：2omotefaul2の40.0秒。緑のボールランプ1個がある画像をB=0と読みました。診断ではball候補の信頼度が0.197273で、採用しきい値に達していません。似た画質・サイズ・配置の失敗画像を集める価値があります。
- **OCRの誤読**：1urafaul1の9.0秒で「1回ウラ」を「10ウラ」と読みました。BSOの矩形注釈とは別の改善課題です。
- **対象条件の限界**：表示が消えない場面は、BSOや走者の注釈を増やしても今回の消失条件では選べません。

既存の学習画像・注釈一式はこの環境で確認できていないため、「学習データが足りないことが原因」とはまだ断定していません。再学習候補を画像で示し、正解確認用CSVとクラス数の集計ツールを用意しています。

## 入力と再現性

- 今回再添付された1urafaul1の転送は32MiB上限に達しました。検証には以前この環境でパーツから復元・保存していた同名10本を使っています。今回の再添付とバイト一致を確認できていないことを明記します。[入力ファイル情報・SHA256](validation/inputs.json) を記録しました。
- `test.mp4` 完全版はこのクラウド環境へ取り込めておらず、今回の実行対象に含みません。手元で実行するコマンドは[README](README.md)にあります。
- 以前受け取ったvideo_info.jsonではtest.mp4は640×360・8075.07秒でした。今回の短い動画より表示が小さくなるため、手元の完全版でも出力画像を確認してください。
- Linux・CPUで実行。VideoMAEエンコーダーはbfloat16、Compact ActionFormerはfloat32。Windows実機の実行は未確認です。設定生成と事前確認は、既存の3環境を明示したLinux側で動作確認しました。
- エンコーダーはMCG-NJU/videomae-base-finetuned-kinetics、revision 488eb9a0565f257b32866000305c8178965eb9f6。投球区間は既存の簡略化Compact ActionFormer重みを使用しました。新しいファウル分類器の学習は行っていません。
- 旧動画全体のVideoMAEキャッシュは使っていません。初回は選ばれた2範囲だけから特徴量を抽出しました。最終の表示・診断情報をそろえる際は、同じ選択範囲の特徴量を再利用して区間推論とレポートを再生成しました。対象範囲・判定理由が変わっていないことを検査しています。
- 全22件の自動テストが通過。10本すべてで[適用範囲の独立検査](validation/scope_verification.json)が通過しました。入力0件ではtorch/transformersの読み込みも行わないことをテストしています。
- [最終化の履歴とコードSHA256](validation/finalization_provenance.json)、[全体集計](validation/totals.json)、[CSV](validation/summary.csv) を保存しました。

結果ZIPの各動画フォルダーにはブラウザー用のindex.html、グラフ、CSV、原始出力、確認画像、注釈確認表を含めています。元動画・学習済み重み・特徴量キャッシュは含めていません。
