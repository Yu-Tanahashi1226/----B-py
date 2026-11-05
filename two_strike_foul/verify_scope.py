"""Independently check that every encoded native clip is inside a selected gate."""
import argparse
import csv
import json
from pathlib import Path

from common import sha256
from pitch_worker import validate_gate, clip_starts


def verify(directory):
    root=Path(directory)
    job=json.loads((root/'job.json').read_text())
    gate=json.loads((root/'gate.json').read_text())
    report=json.loads((root/'pitch_report.json').read_text())
    validate_gate(gate,job)
    if report['gate_sha256']!=sha256(root/'gate.json'):raise ValueError('Gate hash differs from the pitch run')
    if report['whole_video_encoder_used']:raise ValueError('Whole-video encoding is forbidden')
    with (root/'encoded_ranges.csv').open(encoding='utf-8-sig',newline='') as stream:
        ranges=list(csv.DictReader(stream))
    expected={(w['window_id'],start,start+16) for w in gate['windows'] for start in clip_starts(w)}
    actual={(int(r['window_id']),int(r['start_frame']),int(r['end_frame_exclusive'])) for r in ranges}
    if actual!=expected or len(ranges)!=len(expected):raise ValueError('Encoded clips are missing, duplicated or outside the gate')
    if len(ranges)!=report['encoded_clips']:raise ValueError('Clip report count mismatch')
    with (root/'pitch_proposals.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            window=next(w for w in gate['windows'] if w['window_id']==int(row['window_id']))
            if not window['start_sec']<=float(row['start_sec'])<float(row['end_sec'])<=window['end_sec']:
                raise ValueError('A proposal lies outside the selected window')
    intervals=sorted((w['start_frame'],w['end_frame_exclusive']) for w in gate['windows'])
    merged=[]
    for begin,end in intervals:
        if merged and begin<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
        else:merged.append([begin,end])
    covered=sum(end-begin for begin,end in merged)
    return dict(passed=True,selected_windows=len(gate['windows']),encoded_clips=len(ranges),
                selected_video_sec=covered/job['fps'],source_video_sec=job['duration_sec'],
                selected_fraction=covered/job['frames'],outside_gate_clips=0,whole_video_encoder_used=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('result_directory');args=parser.parse_args()
    result=verify(args.result_directory)
    Path(args.result_directory,'scope_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
