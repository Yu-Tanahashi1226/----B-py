# 野球ルールに基づく読取りの保留と補正前後の評価

2026年11月12日更新。YOLO・PaddleOCRの原出力と、時系列で採用する値を分けました。
新しい実行では、OCRの後・VideoMAEの対象選択の前に自動で適用します。
既存の `config.local.json` に `rules` 節がなくても、既定で有効です。

## 今あるtest.mp4の結果だけで表を作る

更新版ZIPの `two_strike_foul` 内のプログラムを、同名フォルダーに配置してください。
既存の `config.local.json`、動画、重み、結果フォルダーは残します。
以下は、以前の解析結果が `for_labelimg\test_reference_results` にある場合です。

```powershell
cd C:\Users\yutan\kenkyuu\for_labelimg\two_strike_foul
python validate_scoreboard.py --observations "..\test_reference_results\analysis\scoreboard\observations.jsonl" --output "..\test_reference_results\rule_review"
Start-Process "..\test_reference_results\rule_review\review.html"
```

このコマンドは保存済みの読取りを再確認するだけです。YOLO・PaddleOCR・VideoMAEを起動せず、原出力も変更しません。
以前の `analysis.csv` や抽出MP4を更新する処理は含みません。
投球ごとにVideoMAEの対象選択が変わるかも比較する場合は、同じコマンドに次を追加します。

```powershell
--events "..\test_reference_results\analysis\verification_job.json"
```

元動画のハッシュが一致する投球時刻だけを使い、`pitch_gate_comparison.csv` を出力します。
これは条件選択の比較であり、ファウル判定の精度評価ではありません。
新しく統合解析を実行するときのコマンドは従来どおりです。補正後の有効な観測を対象選択に使います。

```powershell
python run_foul_pipeline.py --config config.local.json --video "..\test.mp4" --references-dir ".." --output "..\test_reference_results_rules" --scoreboard-scope full
```

`matched` は参考10場面と前後だけ、`full` は動画全体のスコアボードを調べます。

## どの値を採用するか

| 読取り | 記録する状態 | 確認する条件 |
|---|---|---|
| 同じ打者と考えられる場面でBが2→1、Sが2→1 | 直前のB/Sを保持 | 打者交代の根拠がなければ不採用 |
| 同じ回・表裏でOが2→1 | 直前のOを保持 | 回の交代を確認するまで減少を採用しない |
| 同じ試合で得点が1→0 | 直前の得点を保持 | 得点減少として要確認 |
| チーム名が別の文字に変化 | 初めに確認した表記を保持 | 同じ試合・上下段のチーム配置が固定という前提 |
| 未検出・判読不能 | 直前値を保持。初めから不明なら空欄 | 0とは扱わない |
| アウト・出塁・得点増加の直後にB/Sがリセット | 複数回の確認後に採用 | 打者交代の可能性を示す変化が必要 |
| 回・表裏が次へ進み、OとB/Sがリセット | 複数回の確認後に採用 | 回と表裏の組で交代を確認 |
| 回が後戻りする、観測が続くのに大きく飛ぶ | 直前値を保持 | リプレイやOCR誤読等として要確認 |

B/S/Oと走者の変化は既定で2観測、OCRの新しい値は独立したOCR実行で2回確認します。
同じ結果を使い回したOCRキャッシュは、2回目の独立した確認には数えません。
打者交代の根拠は既定で8秒以内とし、古いアウト増加を後のB/S減少の理由に使いません。
回交代直後に全ランプが消えてBSOを読めない場合も、交代を確認した情報を保持し、次に読めたカウントを確認します。読めない間に0を作り出す処理はしません。
回・表裏が大きく飛ぶ読取りは、15秒を超える観測の空白がある場合だけ、複数回確認して再同期できます。

### 保持値をファウルの証拠にしない

たとえば、前が0B・2Sで、次のカウントが未検出だったとします。
記録用には0B・2Sを保持しますが、実際に後の表示を読めたわけではありません。
この状態を「前後でBSOが同じ」とすると、通常の投球までファウル候補にしてしまいます。
そのため、B/S/Oのいずれかを保持した観測は、VideoMAEに進める前後比較に使用しません。
得点・回・走者にも、保持値と新しく確認した値の区別を付けます。

## 出力表

保存済み結果の再確認では指定した `rule_review`、新しい統合解析では `analysis\scoreboard\rules` に作ります。
旧消失条件モードの場合は結果フォルダー内の `rules` です。

| ファイル | 内容 |
|---|---|
| `review.html` | 補正理由と評価表をブラウザーで表示 |
| `scoreboard_before_after.csv` | 全時刻の原読取り `raw_*`、記録値、状態、理由、最後に読めてからの秒数 |
| `corrections.csv` | 保持・未確定になった項目と理由。未検出・確認待ちも含む |
| `metrics.csv` | 同じ正解時刻・項目での補正前後の適合率・再現率・正解率・出力率 |
| `ground_truth_template.csv` | 動画を見て正解を入力する表。正解欄は空欄 |
| `observations.jsonl` | このフォルダー内だけに保存する補正後の状態。元の原出力は変更しない |
| `summary.json` | 不採用フレーム数、理由、使用した設定と原出力のハッシュ |

状態は `observed`＝今回確認できた値、`cached`＝同じ表示区間のOCRキャッシュ、
`held`＝直前から保持した推定値、`unknown`＝まだ確定値がない状態です。
`count_comparison_allowed=False` の行は、BSO不変の証拠には使いません。
不採用の項目数は誤検出数ではありません。たとえばB/Sを組として保留した場合は、BとSの2項目を数えます。

## 適合率・再現率を出す

`ground_truth_template.csv` をExcelで開き、動画を見て `true_*` の列へ正解を入力します。
確認できない項目は空欄にします。`frame`、`video`、`source_sha256` は変更しません。
走者・表示有無は1＝あり、0＝なし、`true_half` は `top` / `bottom` または表 / 裏です。
保存時はCSV UTF-8を選びます。入力済みの正解表は再実行で上書きしません。

```powershell
python validate_scoreboard.py --observations "..\test_reference_results\analysis\scoreboard\observations.jsonl" --output "..\test_reference_results\rule_review" --ground-truth "..\test_reference_results\rule_review\ground_truth_template.csv"
Start-Process "..\test_reference_results\rule_review\review.html"
```

同じ正解フレーム・項目を補正前後で比較します。都合の悪い行を削除して分母を減らしません。
保持値も出力した予測として評価するため、保持によって本当の変化を見逃せば成績が下がります。
ここでの指標は「項目と値」をラベルとする状態認識の分類評価です。
正しく値を出せたらTP、違う値を出せばFPとFN、未出力ならFNに数えます。
適合率＝TP/(TP+FP)、再現率＝TP/(TP+FN)です。
この定義では、全正解項目を分母にした正解率は再現率と同じ値になります。
矩形のIoUに基づくYOLOのAPや、ファウルイベント検出の適合率・再現率とは区別してください。
正解未入力の項目は「未評価」と表示します。ファウル評価には別途、投球ごとの真の結果が必要です。

改善を検証する際は失敗画像だけを選ばず、変更前に決めた評価範囲を全て確認してください。
ルール・閾値を調整した動画と、最終評価に使う別の試合も分けます。

## 設定と限界

`config.local.json` に追加できます。既存の別の設定は残してください。

```json
"rules": {
  "enabled": true,
  "min_stable_samples": 2,
  "ocr_stable_samples": 2,
  "confirmation_window_sec": 6.0,
  "gap_review_sec": 15.0,
  "reset_evidence_max_age_sec": 8.0,
  "initial_state": {},
  "team_aliases": {}
}
```

最初の誤読が連続して採用されると、その誤った状態を保持し続ける場合があります。
その場合は元動画と確認表を調べ、開始時点を目視確認したうえで `initial_state` に
`{"upper_team":"京大","lower_team":"関学"}` のように指定できます。
略称の揺れが分かっている場合だけ `team_aliases` に `{"関西学院":"関学"}` のように設定します。
設定を変更して保存済み結果を再確認するときは、コマンドへ `--config config.local.json` を追加します。

打者交代・回交代の判定も認識結果に依存します。これらを読み落とすと正しいリセットを保留する場合があります。
反対に、ルール上は可能な誤読（本当は1Bなのに2Bなど）は、ルールだけでは除去できません。
この処理はYOLOやOCRの重みを改善するものではありません。
チームの上下段が入れ替わる放送、別試合を連結した映像、リプレイ、公式な得点訂正では、
固定チーム・得点非減少という前提が合わない場合があります。該当箇所を確認し、試合を区切って再実行してください。
「不採用」と「モデルの誤検出確定」は同義ではありません。

## 今回確認した範囲

保存済みの参考10動画・762観測に適用し、30フレームの読取りをルール不整合として保留しました。
これは全てを目視確認して得た誤検出数ではありません。原出力10ファイルのハッシュが変わらないことも確認しました。
以前に目視確認した `2omotefaul2.mp4` の40.0秒では、本当は1Bのところを0Bと読み落としていました。
今回の処理はこの0Bを採用せず、直前の1Bを保持しました。
この1フレームの正解7項目では一致数が6→7になりましたが、既知の失敗例を選んだ動作確認です。
これを全体の適合率・再現率や独立した性能改善として報告しないでください。
実際の5ura内の3投球について、対象選択が補正前後で2件選択・1件除外のままであることも確認しました。
今回test.mp4全体の認識やVideoMAEは再実行していません。手元の保存済み結果は上記コマンドで確認できます。
