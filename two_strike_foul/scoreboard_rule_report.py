"""Before/after tables and ground-truth evaluation of scoreboard STATE values.

Precision/recall here are micro classification metrics over (field, value)
labels, not bounding-box detection AP or foul-event detection metrics.
"""
from collections import Counter
import csv
import html
import json
from pathlib import Path

from common import read_jsonl, save_json, sha256, write_jsonl
from scoreboard_rules import FIELDS, RUNNER_FIELDS, REASONS, filter_observations, flatten, normalize, team_key


LABELS = dict(balls='ボール', strikes='ストライク', outs='アウト', runner_first='一塁走者',
              runner_second='二塁走者', runner_third='三塁走者', inning='回', half='表裏',
              upper_team='上段チーム', lower_team='下段チーム', upper_score='上段得点', lower_score='下段得点',
              board_present='表示の有無', all_fields='全項目')
EVAL_FIELDS = FIELDS + ('board_present',)
REJECT_REASONS = {'invalid_value', 'team_name_changed', 'score_decreased',
                  'outs_decreased_without_inning_change', 'count_decreased_without_batter_change_evidence',
                  'inning_moved_backwards', 'inning_jump_needs_review', 'partial_inning_change',
                  'runner_drop_without_out_or_run_evidence'}


def csv_out(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def video_name(path):
    return str(path).replace('\\', '/').rsplit('/', 1)[-1]


def truth_value(field, value):
    text = str(value).strip()
    if text.casefold() in ('', '?', 'unknown', 'null', '未確認', '不明'):
        return None
    if field in RUNNER_FIELDS or field == 'board_present':
        parsed = {'true': True, 'false': False, '1': True, '0': False}.get(text.casefold())
    elif field in ('upper_team', 'lower_team', 'half'):
        parsed = normalize(field, text)
    else:
        try:
            number = float(text)
            parsed = normalize(field, int(number)) if number.is_integer() else None
        except (ValueError, OverflowError):
            parsed = None
    if parsed is None:
        raise ValueError(f'正解CSVの値が不正です: {field}={value!r}')
    return parsed


def evaluate(raw, corrected, ground_truth, source):
    labels, seen = [], {}
    matched_identity = False
    if ground_truth:
        with Path(ground_truth).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                digest, name = row.get('source_sha256', '').strip(), row.get('video', '').strip()
                if not digest and not name:
                    raise ValueError('正解CSVにはsource_sha256またはvideoが必要です')
                if digest and source.get('sha256') and digest != source['sha256']:
                    if name and video_name(name) == video_name(source.get('video', '')):
                        raise ValueError('同名動画ですが正解CSVのsource_sha256が異なります')
                    continue
                if name and video_name(name) != video_name(source.get('video', '')):
                    continue
                if digest and not source.get('sha256'):
                    raise ValueError('正解CSVの動画ハッシュを確認するjob.jsonがありません')
                matched_identity = True
                frame_number = float(row['frame'])
                if not frame_number.is_integer() or frame_number < 0:
                    raise ValueError('正解CSVのframeは0以上の整数にしてください')
                frame = int(frame_number)
                for field in EVAL_FIELDS:
                    alias = 'true_' + field[7:] if field in RUNNER_FIELDS else 'true_' + field
                    value = truth_value(field, row.get('true_' + field, row.get(alias, '')))
                    if value is None:
                        continue
                    key = frame, field
                    if key in seen and seen[key] != value:
                        raise ValueError(f'正解CSVのラベルが競合しています: {key}')
                    if key not in seen:
                        labels.append((frame, field, value))
                        seen[key] = value
        if not matched_identity:
            raise ValueError('正解CSVに今回の動画と一致する行がありません')

    maps = [{r['frame']: dict(flatten(r), board_present=r.get('board') is not None) for r in data}
            for data in (raw, corrected)]
    results = []
    for index, variant in enumerate(('before', 'after')):
        for field in EVAL_FIELDS + ('all_fields',):
            counts = Counter()
            frames = set()
            for frame, key, truth in labels:
                if field != 'all_fields' and key != field:
                    continue
                frames.add(frame)
                counts['labeled_values'] += 1
                prediction = maps[index].get(frame, {}).get(key)
                if prediction is not None:
                    counts['emitted_values'] += 1
                equal = prediction == truth
                if key.endswith('_team') and prediction is not None:
                    equal = team_key(prediction) == team_key(truth)
                if equal:
                    counts['tp'] += 1
                else:
                    counts['fn'] += 1
                    if prediction is not None:
                        counts['fp'] += 1
            n, emitted, tp = counts['labeled_values'], counts['emitted_values'], counts['tp']
            results.append(dict(variant=variant, field=field, field_ja=LABELS[field],
                                metric_type='state_value_micro', status='evaluated' if n else 'not_evaluated',
                                labeled_frames=len(frames), labeled_values=n, emitted_values=emitted,
                                tp=tp, fp=counts['fp'], fn=counts['fn'],
                                precision=tp / emitted if emitted else None, recall=tp / n if n else None,
                                accuracy=tp / n if n else None, coverage=emitted / n if n else None))
    return results


def write_html(path, summary, audit, metrics):
    esc = lambda v: html.escape('—' if v is None else str(v))
    page = '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>野球ルールによる読取り確認</title><style>body{font:16px/1.7 sans-serif;max-width:1250px;margin:25px auto;padding:16px}table{border-collapse:collapse;width:100%;margin:15px 0}td,th{border:1px solid #ccc;padding:6px;text-align:left}th{background:#edf3f7}.scroll{overflow:auto}a{margin-right:14px}</style>'
    page += '<h1>野球ルールによる読取り確認</h1><p>動画：' + esc(summary['video']) + '</p>'
    page += f'<p>{summary["observations"]}観測。不整合のため採用しなかった読取り：{summary["rejected_field_readings"]}項目（{summary["rejected_frames"]}フレーム）。これは正解と比較した誤検出数ではありません。</p>'
    page += '<p>保持した値は直前の推定状態です。新しく読み取れた値とは区別し、保持だけで「カウント不変のファウル」と判断しません。原YOLO・OCR出力は変更しません。</p>'
    page += '<p><a href="scoreboard_before_after.csv">補正前後の全時刻CSV</a><a href="corrections.csv">保留・不採用の理由CSV</a><a href="metrics.csv">比較指標CSV</a><a href="ground_truth_template.csv">正解入力用CSV</a></p>'
    if not any(m['status'] == 'evaluated' for m in metrics):
        page += '<p><strong>適合率・再現率は未評価です。</strong>ground_truth_template.csvのtrue_*を動画で確認して入力し、--ground-truthを付けて再実行してください。予測値を正解欄に自動入力していません。</p>'
    page += '<h2>正解データとの比較</h2><p>読取り状態の分類指標です。矩形のIoUに基づくYOLOの物体検出APや、ファウル検出の適合率ではありません。同じ正解時刻・項目で補正前後を比較し、未出力は見逃しに含めます。保持値も出力予測として評価します。</p><div class="scroll"><table><tr><th>項目</th><th>正解項目数</th><th>前 適合率</th><th>後 適合率</th><th>前 再現率</th><th>後 再現率</th></tr>'
    for field in ('all_fields',) + EVAL_FIELDS:
        before, after = [next(m for m in metrics if m['field'] == field and m['variant'] == variant) for variant in ('before', 'after')]
        cells = [LABELS[field], before['labeled_values']]
        cells += [f'{r[k]:.1%}' if r[k] is not None else '未評価' for k in ('precision', 'recall') for r in (before, after)]
        page += '<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in cells) + '</tr>'
    page += '</table></div><h2>不整合・保留の例</h2><p>先頭150項目を表示します。全件はCSVにあります。</p><div class="scroll"><table><tr><th>秒</th><th>項目</th><th>読取り</th><th>記録値</th><th>理由</th></tr>'
    selected = [a for a in audit if a['reason'] in REJECT_REASONS]
    for a in (selected or audit)[:150]:
        cells = [f'{a["time_sec"]:.2f}', LABELS[a['field']], a['raw_value'], a['recorded_value'], REASONS[a['reason']]]
        page += '<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in cells) + '</tr>'
    Path(path).write_text(page + '</table></div></html>', encoding='utf-8')


def run_validation(job, output_dir=None, ground_truth=None):
    source = Path(job['output']) / 'observations.jsonl'
    out = Path(output_dir) if output_dir is not None else Path(job['output']) / 'rules'
    if source.resolve() == (out / 'observations.jsonl').resolve():
        raise ValueError('原出力と同じ場所へ補正結果を書き込めません。別のoutputフォルダーを指定してください')
    out.mkdir(parents=True, exist_ok=True)
    cfg = job.get('config', {}).get('rules', {})
    fingerprint = dict(input_sha256=sha256(source), settings=cfg,
                       video=job.get('video'), source_sha256=job.get('sha256'),
                       code={n: sha256(Path(__file__).with_name(n)) for n in ('scoreboard_rules.py', 'scoreboard_rule_report.py')},
                       ground_truth_sha256=sha256(ground_truth) if ground_truth else None)
    summary_file = out / 'summary.json'
    if summary_file.is_file():
        previous = json.loads(summary_file.read_text(encoding='utf-8'))
        if previous.get('fingerprint', {}).get('input_sha256') != fingerprint['input_sha256']:
            raise ValueError('この出力フォルダーは別の観測データ用です。別名のoutputを指定してください')
        required = ('observations.jsonl', 'review.html', 'scoreboard_before_after.csv', 'corrections.csv', 'metrics.csv', 'ground_truth_template.csv')
        if previous.get('fingerprint') == fingerprint and all((out / name).is_file() for name in required):
            return read_jsonl(out / 'observations.jsonl')
    raw = read_jsonl(source)
    if any('rule_validation' in row for row in raw):
        raise ValueError('補正後のファイルではなく、元のYOLO/OCR出力observations.jsonlを指定してください')
    corrected, audit = filter_observations(raw, cfg)
    metrics = evaluate(raw, corrected, ground_truth, job)
    write_jsonl(out / 'observations.jsonl', corrected)
    table = []
    for original, row in zip(raw, corrected):
        a, b = flatten(original), flatten(row)
        info = row.get('rule_validation', {})
        record = dict(frame=row['frame'], time_sec=row['time_sec'],
                      count_comparison_allowed=info.get('count_comparison_allowed'),
                      board_present=row.get('board') is not None)
        for key in FIELDS:
            state = info.get('fields', {}).get(key, {})
            record.update({key: b[key], 'raw_' + key: a[key], key + '_status': state.get('status'),
                           key + '_reason': state.get('reason'), key + '_source_age_sec': state.get('source_age_sec')})
        table.append(record)
    csv_out(out / 'scoreboard_before_after.csv', table, list(table[0]) if table else ['frame', 'time_sec'])
    audit_fields = ['frame', 'time_sec', 'field', 'field_ja', 'raw_value', 'recorded_value', 'status', 'reason', 'reason_ja', 'source_age_sec', 'count_comparison_allowed']
    csv_out(out / 'corrections.csv', [dict(a, field_ja=LABELS[a['field']], reason_ja=REASONS[a['reason']]) for a in audit], audit_fields)
    csv_out(out / 'metrics.csv', metrics, list(metrics[0]))
    template = out / 'ground_truth_template.csv'
    if not template.exists():
        annotations = [dict(video=video_name(job.get('video', '')), source_sha256=job.get('sha256', ''),
                            frame=r['frame'], time_sec=r['time_sec'], **{'true_' + k: '' for k in EVAL_FIELDS}, note='') for r in raw]
        csv_out(template, annotations, list(annotations[0]) if annotations else ['video', 'source_sha256', 'frame', 'time_sec'] + ['true_' + k for k in EVAL_FIELDS] + ['note'])
    rejected = [a for a in audit if a['reason'] in REJECT_REASONS]
    summary = dict(video=video_name(job.get('video', '')), source_sha256=job.get('sha256'), enabled=cfg.get('enabled', True),
                   observations=len(raw), rejected_field_readings=len(rejected), rejected_frames=len({a['frame'] for a in rejected}),
                   reasons=dict(Counter(a['reason'] for a in rejected)),
                   ground_truth=str(ground_truth) if ground_truth else None,
                   evaluation_status='evaluated' if any(m['status'] == 'evaluated' for m in metrics) else 'not_evaluated',
                   metric_scope='state_value_micro; not object detection or foul detection',
                   raw_predictions_preserved=True, held_values_are_new_observations=False, fingerprint=fingerprint)
    save_json(summary_file, summary)
    write_html(out / 'review.html', summary, audit, metrics)
    return corrected
