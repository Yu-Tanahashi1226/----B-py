"""Reprocess saved YOLO/OCR observations without starting any ML model."""
import argparse
import csv
import json
from pathlib import Path
import sys

from common import read_jsonl
from scoreboard_rule_report import csv_out, run_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', required=True, help='保存済みobservations.jsonl')
    parser.add_argument('--output', required=True, help='原出力とは別の確認用フォルダー。同じ入力なら再利用可能')
    parser.add_argument('--config', help='任意。rules節だけを読み、モデルやPython環境は起動しない')
    parser.add_argument('--ground-truth', help='目視で入力した正解CSV。空欄は評価対象外')
    parser.add_argument('--events', help='任意。既存のanalysis/verification_job.jsonで投球条件の補正前後も比較')
    args = parser.parse_args()
    source = Path(args.observations).resolve()
    if source.name != 'observations.jsonl':
        raise ValueError('原YOLO/OCR出力のobservations.jsonlを指定してください')
    job_path = source.with_name('job.json')
    job = json.loads(job_path.read_text(encoding='utf-8-sig')) if job_path.exists() else {}
    job['output'] = str(source.parent)
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        job['config'] = dict(job.get('config', {}), rules=cfg.get('rules', {}))
    rows = run_validation(job, Path(args.output).resolve(), args.ground_truth)
    if args.events:
        from event_count_gate import count_gate
        payload = json.loads(Path(args.events).read_text(encoding='utf-8-sig'))
        if payload.get('source_sha256') != job.get('sha256'):
            raise ValueError('投球時刻の元動画と観測のsource_sha256が一致しません')
        events = sorted(payload['events'], key=lambda e: e['start_sec'])
        raw = read_jsonl(source)
        comparison = []
        for i, event in enumerate(events):
            following = events[i + 1]['start_sec'] if i + 1 < len(events) else None
            a, b = [count_gate(data, event['start_sec'], event['end_sec'], following,
                              job.get('config', {}).get('sample_period_sec', .5)) for data in (raw, rows)]
            comparison.append(dict(event_id=event['event_id'], reference=event.get('reference'),
                                   pitch_index=event.get('pitch_index'), start_sec=event['start_sec'], end_sec=event['end_sec'],
                                   before_selected=a['selected'], after_selected=b['selected'],
                                   before_reason=a['reason'], after_reason=b['reason']))
        csv_out(Path(args.output) / 'pitch_gate_comparison.csv', comparison,
                list(comparison[0]) if comparison else ['event_id', 'before_selected', 'after_selected'])
    print('確認ページ:', Path(args.output).resolve() / 'review.html')
    print('補正前後の表: scoreboard_before_after.csv / 理由: corrections.csv / 指標: metrics.csv')
    if not args.ground_truth:
        print('適合率・再現率は未評価です。正解CSVを入力し、--ground-truthを付けて再実行してください。')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        print('停止:', exc, file=sys.stderr)
        raise SystemExit(1)
