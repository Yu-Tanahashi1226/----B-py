"""YOLO and PaddleOCR workers; each runs inside its own existing environment."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import time

from common import frames_at, metadata, read_jsonl, save_json, write_jsonl
from board_adapter import board_layout_status, inner_state, field_text, parse_inning, parse_score, preprocess_region


def yolo(job):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(job['config']['yolo'].get('threads', 2))
    cfg = job['config']['yolo']
    board_model, inner_model = YOLO(cfg['weights']), YOLO(cfg['inner_weights'])
    if not {'ball', 'strike', 'out', 'runner'} <= {str(n).lower() for n in inner_model.names.values()}:
        raise ValueError(f'BSO/runner用の重みではありません: {inner_model.names}')
    classes = [i for i, name in board_model.names.items() if str(name).lower() == 'scoreboard']
    if not classes:
        raise ValueError('scoreboardクラスがありません。COCOのyolov8n.pt等は使えません')
    output = Path(job['output'])
    started = time.monotonic()
    with (output / 'yolo_boards.jsonl').open('w', encoding='utf-8') as stream:
        for index, (number, frame) in enumerate(frames_at(job['video'], job['sample_frames'])):
            predictions = board_model.predict(frame, conf=cfg['confidence'], classes=classes,
                                               imgsz=cfg['image_size'], device=cfg.get('device', 'cpu'), verbose=False)[0]
            detections = []
            for box in predictions.boxes:
                a, b, c, d = box.xyxy[0].cpu().tolist()
                rect = [max(0, math.floor(a)), max(0, math.floor(b)), min(job['width'], math.ceil(c)), min(job['height'], math.ceil(d))]
                if rect[2] > rect[0] and rect[3] > rect[1]:
                    status = board_layout_status(rect, job['width'], job['height'], cfg['board_layout'])
                    detections.append(dict(box=rect, confidence=float(box.conf[0]), layout_status=status))
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            supported = [d for d in detections if d['layout_status'] == 'supported_corner_layout']
            # Select a supported board even if a wrong-layout box scores higher.
            board = supported[0] if supported else detections[0] if detections else None
            row = dict(frame=number, time_sec=number / job['fps'], board=board, detections=detections,
                       board_layout_status=board['layout_status'] if board else 'no_detection',
                       count=None, runners=None, inner_detections=[], count_status='no_scoreboard')
            if supported:
                a, b, c, d = board['box']
                mx, my = int((c-a)*cfg['inner_margin_rate']), int((d-b)*cfg['inner_margin_rate'])
                rect = [max(0, a-mx), max(0, b-my), min(job['width'], c+mx), min(job['height'], d+my)]
                a, b, c, d = rect
                crop = frame[b:d, a:c]
                inside = inner_model.predict(crop, conf=cfg['inner_confidence'], imgsz=cfg['inner_image_size'],
                                             device=cfg.get('device', 'cpu'), verbose=False)[0]
                row['inner_crop_box'] = rect
                row['inner_detections'] = [dict(box=x.xyxy[0].cpu().tolist(), confidence=float(x.conf[0]),
                                                 label=inner_model.names[int(x.cls[0])]) for x in inside.boxes]
                row['count'], row['runners'], row['count_status'] = inner_state(row['inner_detections'], c-a, d-b)
                row['runner_interpretation'] = 'Predicted occupancy; an undetected runner may be a false negative.'
            elif board:
                row['count_status'] = 'unsupported_layout'
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            if index % 20 == 0:
                print(f'YOLO {index+1}/{len(job["sample_frames"])}', flush=True)
    save_json(output / 'yolo_report.json', dict(sampled_frames=len(job['sample_frames']),
              weights=cfg['weights'], inner_weights=cfg['inner_weights'], elapsed_sec=time.monotonic()-started,
              sampling_note='Whole video, sampled frames. Disappearances shorter than sample period can be missed.'))


def make_ocr(cfg):
    from paddleocr import PaddleOCR
    major = int(importlib.metadata.version('paddleocr').split('.')[0])
    if major < 3:
        raise ValueError('このプログラムはPaddleOCR 3系・指定ローカルモデル用です')
    options = dict(use_doc_orientation_classify=False, use_doc_unwarping=False,
                   use_textline_orientation=False, device=cfg.get('device', 'cpu'),
                   enable_mkldnn=cfg.get('enable_mkldnn', False), cpu_threads=cfg.get('cpu_threads', 2), lang='japan')
    for key, name in [('detector', 'text_detection_model'), ('recognizer', 'text_recognition_model')]:
        options[name + '_name'] = cfg[key + '_name']
        options[name + '_dir'] = cfg[key + '_dir']
    return PaddleOCR(**options)


def recognize(engine, image):
    tokens = []
    for result in engine.predict(image):
        data = result.json
        data = data() if callable(data) else data
        data = json.loads(data) if isinstance(data, str) else data
        data = data.get('res', data)
        polygons = data.get('rec_polys', data.get('dt_polys', []))
        for text, score, polygon in zip(data.get('rec_texts', []), data.get('rec_scores', []), polygons):
            tokens.append(dict(text=str(text), confidence=float(score), polygon=[[float(x), float(y)] for x, y in polygon]))
    return tokens


def crop_region(board, rect):
    h, w = board.shape[:2]
    a, b, c, d = rect
    return board[round(b*h):round(d*h), round(a*w):round(c*w)]


def ocr(job):
    import cv2
    from ocr_fallback import numeric_recognition
    cfg = job['config']['ocr']
    output = Path(job['output'])
    rows = read_jsonl(output / 'yolo_boards.jsonl')
    # Schedule independently for each visible block, so reappearance is always
    # read immediately even when the regular OCR interval has not elapsed.
    scheduled = []
    last = -float('inf')
    previously_visible = False
    previous_count = None
    for row in rows:
        visible = row['board_layout_status'] == 'supported_corner_layout'
        count = row.get('count')
        changed = count != previous_count
        if visible and (not previously_visible or changed or row['time_sec']-last >= cfg['interval_sec']-1e-6):
            scheduled.append(row['frame'])
            last = row['time_sec']
        previously_visible, previous_count = visible, count
        row.update(scoreboard=None, tokens=[], ocr_status='not_scheduled' if visible else 'no_supported_scoreboard')
    selected = {r['frame']: r for r in rows if r['frame'] in set(scheduled)}
    engine = make_ocr(cfg) if scheduled else None
    started = time.monotonic()
    for index, (number, frame) in enumerate(frames_at(job['video'], scheduled)):
        row = selected[number]
        a, b, c, d = row['board']['box']
        board = frame[b:d, a:c]
        values = {}
        fallback = {}
        for field, rect in cfg['regions'].items():
            region = crop_region(board, rect)
            if region.size == 0:
                values[field] = ''
                continue
            if field.endswith('_team'):
                prepared = cv2.resize(region, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            else:
                prepared = preprocess_region(region, yellow=field == 'inning')
            tokens = recognize(engine, prepared)
            for token in tokens:
                token.update(region=field, coordinate_space='preprocessed_field_pixels')
            text, accepted = field_text(tokens, prepared.shape[0], cfg['minimum_confidence'])
            for token in tokens:
                token['accepted_for_field_parse'] = token in accepted
            row['tokens'].extend(tokens)
            values[field] = text
            if field.endswith('_score') and parse_score(text) is None and cfg.get('numeric_recognition_fallback', True):
                fallback[field] = numeric_recognition(engine, prepared)
        inning, half = parse_inning(values.get('inning', ''))
        row['scoreboard'] = dict(inning=inning, half=half, raw=values)
        for field in ['upper_score', 'lower_score']:
            value = parse_score(values.get(field, ''))
            row['scoreboard'][field] = value if value is not None else fallback.get(field, {}).get('accepted_value')
        for field in ['upper_team', 'lower_team']:
            row['scoreboard'][field] = values.get(field) or None
        row.update(ocr_status='inferred', numeric_recognition_fallback=fallback)
        print(f'OCR {index+1}/{len(scheduled)} {number/job["fps"]:.2f}s', flush=True)
    # Context is only propagated inside the SAME visible block, with explicit age.
    # It is never carried through an absent/unsupported board or a BSO transition.
    last_context, last_time, last_count = None, None, None
    for row in rows:
        if row['board_layout_status'] != 'supported_corner_layout' or row.get('count') != last_count:
            last_context, last_time = None, None
        if row['scoreboard'] is not None:
            last_context, last_time = row['scoreboard'], row['time_sec']
            row['context_source_sec'] = last_time
        elif last_context is not None and row['time_sec']-last_time <= cfg['context_max_age_sec']:
            row['scoreboard'] = dict(last_context)
            row['context_source_sec'] = last_time
            row['ocr_status'] = 'same_visible_block_cached'
        last_count = row.get('count')
    write_jsonl(output / 'observations.jsonl', rows)
    save_json(output / 'ocr_report.json', dict(ocr_frames=len(scheduled), elapsed_sec=time.monotonic()-started,
              fields=list(cfg['regions']), model=cfg['recognizer_name'],
              unknown_policy='Unrecognized fields remain null; team strings are OCR observations, not normalized ground truth.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['yolo', 'ocr', 'metadata'])
    parser.add_argument('path')
    args = parser.parse_args()
    if args.stage == 'metadata':
        print(json.dumps(metadata(args.path)))
    else:
        job = json.loads(Path(args.path).read_text(encoding='utf-8-sig'))
        {'yolo': yolo, 'ocr': ocr}[args.stage](job)


if __name__ == '__main__':
    main()
