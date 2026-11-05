"""Native VideoMAE + saved Compact ActionFormer, only on scoreboard-gated windows.

No full-video inference or reuse of old full-video feature/proposal caches is
available here. The gate file must exist before this worker can be called.
"""
import argparse
import csv
import json
from pathlib import Path
import time

from common import save_json, sha256


def clip_starts(window):
    return list(range(window['start_frame'], window['end_frame_exclusive'] - 16 + 1, 8))


def validate_gate(gate, job):
    if gate.get('scope') != 'scoreboard_gated_only':
        raise ValueError('VideoMAE requires a scoreboard-first gate')
    requests = {r['request_id']: r for r in gate['requests']}
    episodes = {e['episode_id']: e for e in gate['episodes']}
    for window in gate['windows']:
        if not 0 <= window['start_frame'] < window['end_frame_exclusive'] <= job['frames']:
            raise ValueError('Gate window is outside the video')
        if window['end_frame_exclusive'] - window['start_frame'] < 16:
            raise ValueError('A VideoMAE window must contain at least 16 native frames')
        if not window['request_ids']:
            raise ValueError('Unrequested inference window')
        for identity in window['request_ids']:
            request = requests[identity]
            parent = episodes[request['parent_episode_id']]
            if not parent['selected'] or parent['reason'] != 'two_strikes_bso_unchanged':
                raise ValueError('Window was not selected by the two-strike scoreboard rule')
            if request['role'] == 'following_pitch_comparison' and not gate['settings']['include_following_pitch']:
                raise ValueError('Following-pitch inference was not enabled')
            if request['window_id'] != window['window_id'] or any(request[k] != window[k] for k in ['start_frame', 'end_frame_exclusive']):
                raise ValueError('Request/window mismatch')
            if request['role'] not in ['foul_candidate', 'following_pitch_comparison']:
                raise ValueError('Unrecognized inference role')


def run(job):
    output = Path(job['output'])
    gate_path = output / 'gate.json'
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    validate_gate(gate, job)
    cfg = job['config']['pitch']
    columns = ['window_id', 'start_sec', 'end_sec', 'model_score', 'score_before_soft_nms']
    ranges = []
    proposals = []
    reports = []
    started = time.monotonic()
    if gate['windows']:
        from collections import deque
        import cv2
        import numpy as np
        import torch
        from transformers import AutoImageProcessor, VideoMAEModel
        import pitch_backend as backend
        torch.set_num_threads(cfg.get('threads', 4))
        backend.DEVICE = torch.device(cfg.get('device', 'cpu'))
        precision = cfg.get('encoder_precision', 'float32')
        if precision not in ['float32', 'bfloat16']:
            raise ValueError('encoder_precision must be float32 or bfloat16')
        detector = backend.CompactActionFormer(input_dim=768).eval().to(backend.DEVICE)
        checkpoint = torch.load(cfg['checkpoint'], map_location=backend.DEVICE, weights_only=True)
        detector.load_state_dict(checkpoint['model'], strict=True)
        checkpoint_hash = sha256(cfg['checkpoint'])
        for window in gate['windows']:
            starts = clip_starts(window)
            identity = dict(schema_version=1, scope='gated_window_only', source_sha256=job['sha256'],
                            fps=job['fps'], source_frames=job['frames'], start_frame=window['start_frame'],
                            end_frame_exclusive=window['end_frame_exclusive'], clip_frames=16, stride_frames=8,
                            encoder=cfg['encoder'], encoder_revision=cfg['revision'], encoder_precision=precision)
            directory = output / 'window_features' / f'{window["window_id"]:04d}'
            directory.mkdir(parents=True, exist_ok=True)
            meta_path = directory / 'meta.json'
            reused = False
            if meta_path.exists() and json.loads(meta_path.read_text()) == identity:
                features = np.load(directory / 'features.npy', allow_pickle=False)
                if not np.array_equal(np.load(directory / 'starts.npy', allow_pickle=False), starts):
                    raise ValueError('Cached selected-window frame starts do not match')
                reused = True
            else:
                if not hasattr(backend, 'videomae'):
                    backend.processor = AutoImageProcessor.from_pretrained(cfg['encoder'], revision=cfg['revision'])
                    backend.videomae = VideoMAEModel.from_pretrained(cfg['encoder'], revision=cfg['revision']).eval().to(backend.DEVICE)
                cap = cv2.VideoCapture(job['video'])
                if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES, starts[0]):
                    cap.release()
                    raise RuntimeError('Cannot seek the selected VideoMAE window')
                ring, pending, batches = deque(maxlen=16), [], []
                clip_index = 0
                try:
                    for number in range(starts[0], starts[-1] + 16):
                        ok, image = cap.read()
                        if not ok:
                            raise RuntimeError(f'Cannot decode native frame {number}')
                        ring.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        if clip_index < len(starts) and number == starts[clip_index] + 15:
                            pending.append(list(ring))
                            clip_index += 1
                            if len(pending) == cfg.get('batch_size', 2):
                                with torch.autocast(backend.DEVICE.type, dtype=torch.bfloat16, enabled=precision == 'bfloat16'):
                                    batches.append(backend.encode_clips(pending).float())
                                pending = []
                                print(f'VideoMAE window {window["window_id"]}: {clip_index}/{len(starts)}', flush=True)
                    if pending:
                        with torch.autocast(backend.DEVICE.type, dtype=torch.bfloat16, enabled=precision == 'bfloat16'):
                            batches.append(backend.encode_clips(pending).float())
                finally:
                    cap.release()
                features = torch.cat(batches).numpy().astype(np.float32)
                np.save(directory / 'features.npy', features)
                np.save(directory / 'starts.npy', np.asarray(starts, dtype=np.int64))
                save_json(meta_path, identity)
            if features.shape != (len(starts), 768) or not np.isfinite(features).all():
                raise ValueError('Invalid selected-window VideoMAE features')
            ranges.extend(dict(window_id=window['window_id'], start_frame=s, end_frame_exclusive=s+16,
                               start_sec=s/job['fps'], end_sec=(s+16)/job['fps'], encoder_cache_reused=reused) for s in starts)
            raw = backend.predict_segments(detector, features)
            kept = backend.soft_nms(raw, sigma=cfg['soft_nms_sigma'], min_score=cfg['minimum_proposal_score'],
                                    max_segments=cfg.get('max_proposals_per_window', 100)).reshape(-1, 3)
            np.save(directory / 'raw_proposals.npy', raw)
            for begin, end, score in kept:
                start_sec = max(window['start_sec'], (window['start_frame'] + float(begin)*8)/job['fps'])
                end_sec = min(window['end_sec'], (window['start_frame'] + float(end)*8)/job['fps'])
                if end_sec > start_sec and score >= cfg['minimum_proposal_score']:
                    matched = raw[np.isclose(raw[:, 0], begin) & np.isclose(raw[:, 1], end)]
                    proposals.append(dict(window_id=window['window_id'], start_sec=start_sec, end_sec=end_sec,
                                          model_score=float(score), score_before_soft_nms=float(matched[:, 2].max())))
            reports.append(dict(window_id=window['window_id'], feature_shape=list(features.shape), cache_reused=reused,
                                raw_proposals=len(raw), retained_proposals=len(kept)))
    else:
        checkpoint_hash = None  # No torch/transformers import, model load or inference for an empty gate.
    with (output / 'pitch_proposals.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(sorted(proposals, key=lambda r: (r['start_sec'], -r['model_score'])))
    with (output / 'encoded_ranges.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['window_id', 'start_frame', 'end_frame_exclusive', 'start_sec', 'end_sec', 'encoder_cache_reused'])
        writer.writeheader()
        writer.writerows(ranges)
    save_json(output / 'pitch_report.json', dict(scope='scoreboard_gated_only', gate_sha256=sha256(gate_path),
              windows=reports, encoded_clips=len(ranges), proposals=len(proposals), checkpoint_sha256=checkpoint_hash,
              whole_video_encoder_used=False, elapsed_sec=time.monotonic()-started,
              interpretation='VideoMAE features + simplified Compact ActionFormer detect pitch-like motion. This is not a foul-trained classifier.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    args = parser.parse_args()
    run(json.loads(Path(args.job).read_text(encoding='utf-8-sig')))
