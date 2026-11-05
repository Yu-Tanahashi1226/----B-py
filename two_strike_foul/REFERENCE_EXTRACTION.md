# test.mp4から参考10場面を自動で書き出す

今回の目的は、**提供された10本と同じ場面を全体動画から探し、MP4として保存すること**です。映像照合による場面の発見と、BSO・VideoMAEによる投球結果の確認を分けました。

## なぜ前の方法では取りこぼしたか

1. **スコアボードの消失を必須にしていたためです。** 実際には、2omotefaul2の2球目や2urafaul3の1球目後も表示が残っていました。参考10本で前の条件に一致したのは2場面でした。
2. **投球単位の時刻を決める仕組みが足りませんでした。** 表示が残ると、「いつの投球の前後」を比べるかを消失時刻から決められません。新しい方法では、参考動画の映像と24投球の相対時刻を全体動画へ対応付けます。
3. **読み落としを実際のカウント変化として扱っていました。** 2omotefaul2の40秒のボール1個→0個、5uraの118秒のアウト1個→0個を画像で確認しています。これだけで学習データ不足とは断定できませんが、追加注釈を検討する具体例です。
4. **元のVideoMAE＋Compact ActionFormerは投球動作の検出用です。** 投球スコア0.95はファウル確率95%という意味にはなりません。今回、投球検出と別に試作ファウル判別器を加えました。
5. **前の版には、10場面を全体動画の中から照合して音声付きMP4に書き出す処理がありませんでした。** 今回はその処理を追加しています。

## 新しい処理の順序

1. 参考動画の複数の時刻と、全体動画の映像を照合し、10場面の開始時刻を探します。
2. 一意に照合できた場面を `scenes/` に音声付きMP4で保存します。BSOが読めない場合も、照合した場面は残します。
3. YOLOでBSO・走者、PaddleOCRで回・表裏・チーム名・得点を記録します。
4. 各投球の前後について「投球前が2ストライク」「BSOが変わらない」を調べます。**表示の消失は必須にしません。** 読めない値は不明として保留します。
5. この条件に一致した投球だけ、VideoMAEの特徴量、Compact ActionFormerの投球スコア、試作ファウル判別スコアを計算します。
6. 条件と両モデルが支持した投球を `analysis/foul_candidates/` に切り出します。判定には目視確認が必要です。

10本の参考動画にはファウル後の投球も含まれます。`scenes/` は参考動画と同じ長さの10場面、`analysis.csv` はその中の24投球の確認表です。試作の参考注釈では、カウントが増えないファウルは15投球、その他は9投球です。ファウルフライの捕球によるアウトは後者に含めています。

## Windowsでの実行

ZIP内の `two_strike_foul` フォルダーを `for_labelimg` の直下へ置いてください。10本の元の参考動画が `for_labelimg` にある例です。

```powershell
cd C:\Users\yutan\kenkyuu\for_labelimg\two_strike_foul
```

以前の `config.local.json` があればそのまま使えます。なければ既存環境・モデルを探して設定します。

```powershell
python configure_windows.py --project ".."
```

**参考場面の書き出しと、BSO・VideoMAEによる確認を一度に実行：**

```powershell
python run_foul_pipeline.py --config config.local.json --video "..\test.mp4" --references-dir ".." --output "..\test_reference_results" --scoreboard-scope matched
```

`matched` は照合できた場面と前後だけでYOLO・OCRを実行する設定です。以前の目的と同じく動画全体のスコアボードを記録する場合は、最後を `--scoreboard-scope full` にします。全体へのVideoMAE実行にはなりません。全体へのYOLO・OCRは長時間かかり得ます。

```powershell
python open_results.py "..\test_reference_results"
```

確認ページで、元動画を選び直さずに書き出したMP4を再生できます。「BSO・VideoMAEによる確認結果」から投球別の表と動画へ進めます。

### まず切り出しだけ試す場合

```powershell
python extract_reference_scenes.py --config config.local.json --video "..\test.mp4" --references-dir ".." --output "..\test_scenes_only"
```

この場合、YOLO・OCR・VideoMAEの推論は行いません。照合・切り出しはOpenCV、NumPy、FFmpegを使います。

FFmpegが見つからない場合は、表示されたコマンドで `imageio-ffmpeg` を**処理に使うPython環境**へ入れてください。システムのFFmpegがあれば追加インストールは不要です。手動で場所を指定する場合は、`extract_reference_scenes.py` の `--ffmpeg "実際の保存先\ffmpeg.exe"` が使えます。

### 再開する

```powershell
python extract_reference_scenes.py --config config.local.json --video "..\test.mp4" --references-dir ".." --output "..\test_reference_results" --analyze --scoreboard-scope matched --resume
```

設定は前回と同じにします。照合用の索引、書き出し済みMP4、整合するYOLO・OCRの出力、選択された範囲のVideoMAE特徴量を再利用します。モデルの判別は再計算します。元動画や参考動画は変更しません。

## 出力

| 保存先 | 内容 |
|---|---|
| `index.html` | 抽出した場面の動画と、分析結果へのリンク |
| `scenes/*_from_source.mp4` | 参考動画と一致した場面。境界の取りこぼしを避けるため前後0.3秒付き |
| `matches.csv` / `matches.json` | 全体動画内の開始・終了時刻、照合値、重複・未発見の状態 |
| `analysis/index.html` | 各投球のBSO条件と、投球・試作ファウル判別スコア |
| `analysis/analysis.csv` | 24投球のうち照合できた場面に含まれる投球の確認表 |
| `analysis/scoreboard.csv` | BSO・走者・回・表裏・チーム名・得点 |
| `analysis/foul_candidates/*.mp4` | BSO条件・投球検出・試作ファウル判別が支持した投球。要確認 |
| `analysis/scoreboard/annotation_review/` | 読み落としを調べる画像と目視確認用CSV |
| `analysis/encoded_ranges.csv` | VideoMAEに渡した元動画のフレーム範囲 |
| `analysis/verification_report.json` | 対象投球数、入力窓数、対象外への適用がないことの確認 |

照合値は画像の一致度、投球スコアは投球らしさ、試作ファウルスコアは小規模な判別器の出力です。3種類の値を混同しないでください。

## BSOの扱い

- 投球の前5秒までと、後0.5〜8秒の範囲で安定値を探します。次の投球が始まる場合は、その前で区切ります。
- 読み取れるBSOが2回以上連続して一致することを求めます。再表示は要求しません。
- 後からボール・ストライク・アウトの増加やリセットが現れれば、古い表示が一時的に残っていても対象から外します。
- 同じ回・表裏が明示的に読め、B/Sが同じで、アウトだけ一時的に減り、最後に元のBSOへ3回以上安定して戻った場合は、アウトの読み落としが疑われます。その不整合を比較から外して記録します。**元の認識値を書き換えません。** 同じ回と確認できない場合や、元の値へ戻らない場合にはこの処理をしません。
- BSOを確認できない投球を、非ファウルとは判断しません。照合した元の場面を残し、判定を保留します。

## 試作ファウル判別器の限界

元の投球区間検出器に加え、VideoMAEの投球付近〜投球後の特徴を使う小さなロジスティック回帰を作成しました。**参考動画に対する過去の近似的な目視注釈**を使い、15ファウル・9非ファウルの計24投球で検証しています。ユーザーが独立に確定した正解ではありません。

判定する参考動画そのものを学習から除外し、残り9本で学習する方法では、参考注釈との一致は **17/24** でした。内訳はファウル15投球中12をファウル判定、非ファウル9投球中4もファウルと誤判定しています。別試合への性能や、ファウルの確定判定を保証できる結果ではありません。

そのため `confirmed_foul` は空欄、`review_required` はTrueにしています。モデルが支持した投球も確認用候補として保存します。参考動画にない未知のファウルを網羅的に見つける処理ではなく、今回合意した**既知の10場面を探して確認する処理**です。

24投球のラベルを見直す表は `reference_labels_review.csv` です。ファウル判別を改善するための注釈は、BSOの矩形注釈とは別に「投球区間」「カウントが増えないファウルか、それ以外か」の区別が必要です。異なる試合や、非ファウル・見送り・空振り・インプレーも含む例を増やして分離評価します。

モデルの再現・ラベル確認後の再学習には `train_reference_outcome.py` と `reference_training_vectors.npz` を用意しました。再学習時だけscikit-learnが必要です。実行時の判別はNumPyで行います。

```powershell
python train_reference_outcome.py --labels-review reference_labels_review.csv --output reference_outcome_heads.reviewed.json
```

再学習モデルを解析に使うには、比較用の新しいプログラムフォルダーで `reference_outcome_heads.json` として配置し、新しい結果フォルダーで実行してください。元の判別器と結果を保持して比較します。

## 検証範囲

[新しい検証記録](REFERENCE_VALIDATION.md) に、640×360へ縮小・再圧縮した10場面の照合、実際の5uraからの切り出し、BSO・VideoMAEの実行結果を記録しています。`test.mp4` 完全版はクラウドへ取り込めていないため、そこから10本を実際に書き出した結果とは主張しません。

映像照合は同じ映像が含まれることを前提とします。全く別の試合、左右反転、速度変更、途中編集が異なる動画、参考場面の一部しか入っていない場合は、そのまま対応できないことがあります。一致が弱い場合や同じ場面の重複は保留し、数を10に合わせて出力しません。
