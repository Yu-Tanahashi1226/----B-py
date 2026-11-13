"""Put saved YOLO/OCR/VideoMAE results beside editable human labels; no ML rerun."""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import sys

from common import read_jsonl, save_json, sha256
from scoreboard_rules import filter_observations, flatten
from scoreboard_rule_report import EVAL_FIELDS, LABELS, csv_out, video_name, truth_value

ROOT = Path(__file__).resolve().parent
STATE_FIELDS = EVAL_FIELDS
MODEL_FIELDS = ('pitch_present', 'pitch_start_sec', 'pitch_end_sec', 'foul', 'two_strike_foul')
MODEL_LABELS = dict(pitch_present='投球動作がある', pitch_start_sec='投球開始（秒）',
                    pitch_end_sec='投球終了（秒）', foul='ファウルである',
                    two_strike_foul='統合結果：2ストライク後のファウル候補')
STATUS = dict(model_supported_foul_candidate='モデルが支持するファウル候補（要確認）',
              model_did_not_support_foul_review='試作モデルがファウルを支持しない（要確認）',
              inferred_two_strike_foul='条件と投球モデルによるファウル推定（要確認）',
              comparison_pitch_detected='比較用の投球を検出', pitch_not_supported='投球の十分な検出なし',
              before_bso_unknown='前のBSOが不明', after_bso_unknown='後のBSOが不明',
              not_two_strikes='前が2ストライクではない', bso_changed_or_delayed_update='BSO変化・表示更新',
              inning_or_half_changed_review='回・表裏が変化', insufficient_native_video_context='映像が不足')


def number(value):
    if value is None or value == '':
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def state_values(row):
    return dict(flatten(row), board_present=row.get('board') is not None)


def load_labels(path, key, source):
    """Only manual columns are carried forward, never copied from predictions."""
    if not path.is_file():
        return {}
    values = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            digest, name = row.get('source_sha256', ''), row.get('video', '')
            if digest and digest != source['sha256']:
                raise ValueError(f'{path.name}: 別の元動画の正解が含まれています')
            if name and video_name(name) != source['video']:
                raise ValueError(f'{path.name}: 動画名が一致しません')
            label = {k: v for k, v in row.items() if (k.startswith('true_') or k == 'note') and v not in ('', None)}
            for field, value in list(label.items()):
                if field == 'note':
                    continue
                name = field.removeprefix('true_')
                for side in ('before_', 'after_'):
                    name = name.removeprefix(side)
                if name in STATE_FIELDS:
                    value = truth_value(name, value)
                elif name in ('pitch_present', 'foul', 'two_strike_foul'):
                    value = truth_value('board_present', value)
                elif name in ('pitch_start_sec', 'pitch_end_sec'):
                    value = number(value)
                    if value is None or value < 0:
                        raise ValueError(f'{path.name}: 目視の投球時刻が不正です')
                else:
                    raise ValueError(f'{path.name}: 未対応の正解列です: {field}')
                label[field] = '' if value is None else '1' if value is True else '0' if value is False else str(value)
            if not label:
                continue
            identity = str(row[key])
            old = values.setdefault(identity, {})
            if any(k in old and old[k] != v for k, v in label.items()):
                raise ValueError(f'{path.name}: {key}={identity} の正解が競合しています')
            old.update(label)
    return values


def generate_review(results, output=None):
    results = Path(results).resolve()
    analysis = results / 'analysis' if (results / 'analysis' / 'scoreboard' / 'observations.jsonl').is_file() else results
    board = analysis / 'scoreboard' if (analysis / 'scoreboard' / 'observations.jsonl').is_file() else analysis
    raw_path = board / 'observations.jsonl'
    if not raw_path.is_file():
        raise ValueError('保存済みのobservations.jsonlが見つかりません。解析結果フォルダーを指定してください')
    job_path = board / 'job.json'
    if not job_path.is_file():
        raise ValueError('動画を照合するjob.jsonがありません。元の解析結果フォルダーを指定してください')
    job = json.loads(job_path.read_text(encoding='utf-8-sig'))
    source = dict(video=video_name(job['video']), sha256=job.get('sha256'))
    if not source['sha256']:
        raise ValueError('job.jsonに元動画のsha256がありません')
    raw = read_jsonl(raw_path)
    if any('rule_validation' in r for r in raw):
        raise ValueError('補正後のファイルではなく元の解析結果フォルダーを指定してください')
    frames = [r['frame'] for r in raw]
    times = [r['time_sec'] for r in raw]
    if len(set(frames)) != len(frames) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('観測のフレーム・時刻に重複または順序の不整合があります')
    raw_hash = sha256(raw_path)
    corrected = None
    correction_source = '今回の表作成時にルールを適用'
    for directory in (board / 'rules', results / 'rule_review'):
        summary = directory / 'summary.json'
        if summary.is_file() and (directory / 'observations.jsonl').is_file():
            meta = json.loads(summary.read_text(encoding='utf-8-sig'))
            if meta.get('fingerprint', {}).get('input_sha256') == raw_hash:
                candidate = read_jsonl(directory / 'observations.jsonl')
                if [r['frame'] for r in candidate] != frames:
                    raise ValueError('補正結果のフレームが原観測と一致しません')
                corrected, correction_source = candidate, str(directory)
                break
    if corrected is None:
        corrected, _ = filter_observations(raw, job.get('config', {}).get('rules', {}))
    out = Path(output).resolve() if output is not None else analysis / 'manual_review'
    if out == board or out == analysis:
        raise ValueError('確認表は元の解析結果とは別のmanual_reviewフォルダーに保存してください')
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'source.json'
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        if previous.get('source_sha256') != source['sha256'] or previous.get('observations_sha256') != raw_hash:
            raise ValueError('この確認フォルダーは別の観測用です。別の--outputを指定してください')
    labels = load_labels(out / 'scoreboard_review.csv', 'frame', source)
    for parent in (board / 'rules', results / 'rule_review'):
        for key, value in load_labels(parent / 'ground_truth_template.csv', 'frame', source).items():
            existing = labels.setdefault(key, {})
            for field, truth in value.items():
                if field in existing and existing[field] != truth:
                    raise ValueError(f'frame={key}: 既存の正解表同士で値が異なります')
                existing[field] = truth
    state_rows, compact = [], []
    for original, recorded in zip(raw, corrected):
        a, b = state_values(original), state_values(recorded)
        row = dict(video=source['video'], source_sha256=source['sha256'], frame=original['frame'], time_sec=original['time_sec'])
        states = recorded.get('rule_validation', {}).get('fields', {})
        for field in STATE_FIELDS:
            row.update({'pred_' + field: a[field], 'recorded_' + field: b[field], 'true_' + field: '',
                        field + '_status': states.get(field, {}).get('status', 'observed')})
        row['note'] = ''
        row.update(labels.get(str(row['frame']), {}))
        state_rows.append(row)
        compact.append([row['frame'], row['time_sec'], [a[f] for f in STATE_FIELDS],
                        [b[f] for f in STATE_FIELDS], [row[f + '_status'] for f in STATE_FIELDS],
                        (original.get('scoreboard') or {}).get('raw', {})])
    event_path = analysis / 'events.json'
    events = json.loads(event_path.read_text(encoding='utf-8-sig')) if event_path.is_file() else []
    if not isinstance(events, list):
        raise ValueError('events.jsonは投球結果の配列である必要があります')
    pitch_labels = load_labels(out / 'pitch_review.csv', 'event_id', source)
    pitch_rows, event_data = [], []
    identifiers = set()
    for event in events:
        event_id = str(event.get('event_id', event.get('request_id', '')))
        if not event_id or event_id in identifiers:
            raise ValueError('投球結果のevent_idが空または重複しています')
        identifiers.add(event_id)
        gate = event.get('count_gate', event)
        snapshots = {}
        row = dict(video=source['video'], source_sha256=source['sha256'], event_id=event_id,
                   reference=event.get('reference', ''), pitch_index=event.get('pitch_index', ''),
                   window_start_sec=event.get('start_sec'), window_end_sec=event.get('end_sec'),
                   status=event.get('status'), status_ja=STATUS.get(event.get('status'), event.get('status')),
                   count_gate_reason=gate.get('reason', ''),
                   pitch_model_score=event.get('pitch_model_score', event.get('model_score')),
                   foul_model_score=event.get('foul_model_score'))
        for side in ('before', 'after'):
            summary = gate.get(side) or {}
            target = number(summary.get('last_sec'))
            index = bisect.bisect_left(times, target) if target is not None else len(times)
            # Use an actual observation from the summarized run. Never borrow a
            # nearby frame from another pitch when the required sample is absent.
            index = index if index < len(times) and abs(times[index] - target) < 1e-5 else None
            snapshots[side] = index
            row[side + '_frame'] = raw[index]['frame'] if index is not None else None
            row[side + '_time_sec'] = raw[index]['time_sec'] if index is not None else None
            a = state_values(raw[index]) if index is not None else {}
            b = state_values(corrected[index]) if index is not None else {}
            for field in STATE_FIELDS:
                row[side + '_pred_' + field] = a.get(field)
                row[side + '_recorded_' + field] = b.get(field)
                row['true_' + side + '_' + field] = labels.get(str(row[side + '_frame']), {}).get('true_' + field, '')
        inferred_pitch = event.get('pitch_supported_by_model')
        if inferred_pitch is None and row['pitch_model_score'] is not None:
            inferred_pitch = float(row['pitch_model_score']) >= job.get('config', {}).get('pitch', {}).get('pitch_confirmation_score', .5)
        predictions = dict(pitch_present=inferred_pitch, pitch_start_sec=event.get('model_start_sec'),
                           pitch_end_sec=event.get('model_end_sec'), foul=event.get('foul_model_decision'),
                           two_strike_foul=event.get('status') in ('model_supported_foul_candidate', 'inferred_two_strike_foul'))
        for field in MODEL_FIELDS:
            row['pred_' + field] = predictions[field]
            row['true_' + field] = ''
        row['note'] = ''
        row.update(pitch_labels.get(event_id, {}))
        pitch_rows.append(row)
        event_data.append(dict(id=event_id, reference=row['reference'], pitch_index=row['pitch_index'],
                               start=row['window_start_sec'], end=row['window_end_sec'],
                               status=row['status_ja'], pitch_score=row['pitch_model_score'], foul_score=row['foul_model_score'],
                               snapshots=snapshots, predictions=predictions))
    state_labels = {str(r['frame']): {k: v for k, v in r.items() if (k.startswith('true_') or k == 'note') and v != ''} for r in state_rows}
    for row in pitch_rows:
        for side in ('before', 'after'):
            key = row[side + '_frame']
            if key is None:
                continue
            target = state_labels.setdefault(str(key), {})
            for field in STATE_FIELDS:
                value = row['true_' + side + '_' + field]
                if value == '':
                    continue
                true_key = 'true_' + field
                if true_key in target and target[true_key] != value:
                    raise ValueError(f'frame={key}: 投球表とスコアボード表の正解が異なります')
                target[true_key] = value
    for row in state_rows:
        row.update(state_labels.get(str(row['frame']), {}))
    # Preserve handwritten files; refreshed exports are downloaded from the page.
    for path, rows, columns in ((out / 'scoreboard_review.csv', state_rows, list(state_rows[0]) if state_rows else ['video', 'source_sha256', 'frame', 'time_sec']),
                                (out / 'pitch_review.csv', pitch_rows, list(pitch_rows[0]) if pitch_rows else ['video', 'source_sha256', 'event_id'] + ['true_' + f for f in MODEL_FIELDS])):
        if not path.exists():
            csv_out(path, rows, columns)
    event_labels = {r['event_id']: {k: v for k, v in r.items() if (k in ['true_' + f for f in MODEL_FIELDS] or k == 'note') and v != ''} for r in pitch_rows}
    payload = dict(source=source, observations_sha256=raw_hash, correction_source=correction_source,
                   fields=list(STATE_FIELDS), labels=LABELS, model_fields=list(MODEL_FIELDS), model_labels=MODEL_LABELS,
                   observations=compact, events=event_data, state_columns=list(state_rows[0]) if state_rows else [],
                   pitch_columns=list(pitch_rows[0]) if pitch_rows else [], pitch_templates=pitch_rows,
                   state_labels=state_labels, event_labels=event_labels)
    # Embedded OCR text is data. Escape '<' so it cannot terminate the JSON script.
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c').replace('&', '\\u0026')
    template = (ROOT / 'manual_review.html').read_text(encoding='utf-8')
    page = template.replace('<!--REVIEW_DATA-->', '<script id="review-data" type="application/json">' + encoded + '</script>')
    page = page.replace('<!--REVIEW_SCRIPT-->', '<script>' + (ROOT / 'manual_review.js').read_text(encoding='utf-8') + '</script>')
    (out / 'index.html').write_text(page, encoding='utf-8')
    save_json(manifest_path, dict(source_sha256=source['sha256'], observations_sha256=raw_hash,
                                 video=source['video'], observations=len(raw), events=len(events),
                                 correction_source=correction_source, models_rerun=False,
                                 initial_truth_is_blank=not any(v for labels in (state_labels, event_labels) for row in labels.values() for k, v in row.items() if k.startswith('true_')),
                                 human_files_overwritten=False))
    (out / 'README.txt').write_text('index.htmlをブラウザーで開き、手元の元動画を選んで確認します。目視正解はモデル値の右隣へ入力し、最後に「スコアボードCSV保存」「投球CSV保存」を押してください。ブラウザーの自動保存だけに頼らずCSVを保存してください。CSVには予測と正解を並べています。予測値を正解へ自動コピーしません。元の解析結果・動画は変更しません。\n', encoding='utf-8')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, help='test_reference_resultsまたはtest_reference_results_rules等の結果フォルダー')
    parser.add_argument('--output', help='省略時はanalysis/manual_review。既存の手書きCSVは保持')
    args = parser.parse_args()
    out = generate_review(args.results, args.output)
    print('目視の正解を隣に入力:', out / 'index.html')
    print('Excel用: scoreboard_review.csv / pitch_review.csv')
    print('YOLO・OCR・VideoMAEの再実行はありません。最後にページ上のCSV保存ボタンで入力結果を保存してください。')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        print('停止:', exc, file=sys.stderr)
        raise SystemExit(1)
