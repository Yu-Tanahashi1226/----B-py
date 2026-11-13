# 検出結果の隣に、目視の正解を書く

YOLO・PaddleOCR・VideoMAEの出力と、目視した正解を並べた確認ページです。
HTMLの入力欄でも、Excelで開けるCSVでも記録できます。
動画・モデルの再実行は不要です。元の解析結果は変更しません。

## 保存済みtest.mp4の結果から作る

更新用ZIPを `for_labelimg` に解凍し、既存の `two_strike_foul` のプログラムを更新してください。
既存の設定・重み・結果を残して更新できます。
PowerShellで次を実行します。

```powershell
cd C:\Users\yutan\kenkyuu\for_labelimg\two_strike_foul
python make_manual_review.py --results "..\test_reference_results_rules"
Start-Process "..\test_reference_results_rules\analysis\manual_review\index.html"
```

以前の `test_reference_results` を使う場合は、上の2行の `test_reference_results_rules` を `test_reference_results` に変えます。
新しい全体解析が終わる前なら、完了済みの以前の結果で確認ページを作れます。
VideoMAEの完了結果がまだないときは、YOLO・OCRの表だけを作ります。
解析完了後に同じコマンドで再作成すると、投球結果が追加されます。
今後の新しい解析では、解析ページの「検出結果の隣に目視の正解を入力する」からも開けます。

## HTMLで入力する

1. 上部の「手元の元動画を選ぶ」で、解析に使った `test.mp4` を選びます。
2. 「投球ごと：全モデル＋目視」で、前後のスコアボードとVideoMAEの結果を確認します。
3. 各行の黄色い「目視正解」欄へ、正しい値を入力します。
4. 「全時刻：YOLO・OCR＋目視」では、時刻範囲とBSO・走者・文字の表示項目を選べます。
5. 最後に「スコアボードCSV保存」「投球CSV保存」を押します。

たとえば、ボールの行は「読取り0」「補正後1」「目視正解［入力欄］」と並びます。
正解が1なら入力欄に1を書きます。読取り・補正後は変更しません。
未確認は空欄にし、0と分けます。走者・表示有無・ファウルは「あり／はい」「なし／いいえ」を選べます。
同じフレームの正解は、投球表と全時刻表で共有します。
投球前後の代表フレームが取れていない場合は、全時刻表で必要な時刻を確認してください。

入力は可能なブラウザーでは自動保存しますが、**最後に必ずCSV保存ボタンでファイルにも保存してください。**
通常はダウンロードフォルダーへ保存されます。同名のファイルがあると `(1)` などが付く場合があります。
他のパソコンやブラウザーへ移るときは、「入力済みCSVを読み込む」で保存したCSVを選びます。
動画はパソコン内で再生し、サーバーへ送りません。

## Excelで入力する

作成された `analysis\manual_review` フォルダーに、最初から次のCSVを置いています。

| CSV | 内容 |
|---|---|
| `scoreboard_review.csv` | 各時刻のYOLO・OCRの読取り、ルール補正後、その隣の `true_*` 正解欄 |
| `pitch_review.csv` | 投球前後のスコアボード、VideoMAEの投球区間・スコア・試作判別、目視正解欄 |

`true_*` と `note` に入力します。動画名・ハッシュ・フレーム番号・投球番号は変更しません。
BSO・回・得点は数値、走者とファウルは1＝あり、0＝なし、表裏は `top`＝表、`bottom`＝裏です。
投球区間の秒数には小数を入力できます。保存形式はCSV UTF-8です。

入力後に確認ページを再作成すると、CSVの正解も読み込みます。
既存の入力済みCSVは上書きしません。再作成後の最新の予測と正解をまとめて保存するには、ページのCSV保存ボタンを使います。
スコアボードCSVと投球CSVの両方に同じフレームの正解があり、値が異なる場合は、不一致を修正してください。

## BSO・OCRの適合率・再現率に使う

HTMLから保存した最新の `scoreboard_review.csv` を `analysis\manual_review` に置きます。
このCSVには正解と予測が入っていますが、評価コマンドは `true_*` だけを正解として読みます。

```powershell
python validate_scoreboard.py --observations "..\test_reference_results_rules\analysis\scoreboard\observations.jsonl" --output "..\test_reference_results_rules\rule_review" --ground-truth "..\test_reference_results_rules\analysis\manual_review\scoreboard_review.csv"
```

正解を入力した同じ時刻・項目で補正前後を比較します。
全項目を入力する必要はありませんが、確認できた項目だけの成績であることを示してください。
VideoMAEの目視正解は投球CSVに記録します。上のコマンドはBSO・文字情報などの状態認識の評価です。
投球一覧の外にある見逃しや、未知のファウルまで含む動画全体の再現率を、参考24投球だけから算出しないでください。

## 動作確認

既存の実動画5uraの保存済み結果で、146観測・3投球の入力ページを作成しました。
初期正解欄は全て空欄です。モデルの値を正解へ自動コピーしません。
入力したCSVの保持、動画の識別、代表時刻がない場合の扱い、HTMLへのOCR文字の埋込み、既存評価コマンドとの互換性を自動テストで確認しています。
ブラウザーの試験で入力する値は操作確認専用で、研究の正解データや性能評価には使いません。
