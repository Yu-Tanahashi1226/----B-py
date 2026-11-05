"""Small dependency-free helpers shared by the three Python environments."""
import hashlib
import json
from pathlib import Path


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def frames_at(video, samples):
    """Decode only requested frames, sequentially within each requested run."""
    import cv2
    if not samples:
        return
    if samples != sorted(set(samples)):
        raise ValueError('Sample frames must be sorted and unique')
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f'動画を開けません: {video}')
    try:
        next_frame = samples[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
        for number in samples:
            while next_frame <= number:
                if not cap.grab():
                    raise RuntimeError(f'フレームを読めません: {next_frame}')
                next_frame += 1
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError(f'フレームを取得できません: {number}')
            yield number, frame
    finally:
        cap.release()


def metadata(video):
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f'動画を開けません: {video}')
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if frames < 16 or fps <= 0:
            raise ValueError('動画には16フレーム以上と正のFPSが必要です')
        return dict(frames=frames, fps=fps, duration_sec=frames / fps,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
