# Model definitions extracted unchanged from the supplied pitch_extraction.ipynb.
import math
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
DEVICE = torch.device("cpu")
CLIP_FRAMES = 16
FEATURE_STRIDE_FRAMES = 8
FEATURE_BATCH_SIZE = 2
MAX_SEQ_LEN = 256
FPN_STRIDES = [1, 2, 4, 8]

def read_clip(cap: cv2.VideoCapture, start_frame: int, clip_frames: int) -> list[np.ndarray]:
    """start_frameから連続フレームをRGBで読み込む。不足分は最終フレームで埋める。"""
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    for _ in range(clip_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not frames:
        raise RuntimeError(f'フレームを読めません: {start_frame}')
    while len(frames) < clip_frames:
        frames.append(frames[-1].copy())
    return frames

@torch.inference_mode()
def encode_clips(batch_frames: list[list[np.ndarray]]) -> torch.Tensor:
    """複数クリップをVideoMAEで処理し、[B, hidden_size]を返す。"""
    inputs = processor(batch_frames, return_tensors='pt')
    pixel_values = inputs['pixel_values']
    if pixel_values.ndim != 5:
        raise RuntimeError(f'VideoMAE入力が5次元ではありません: {tuple(pixel_values.shape)}')
    if pixel_values.shape[1] != CLIP_FRAMES or pixel_values.shape[2] != 3:
        raise RuntimeError(f'VideoMAE入力形状が不正です: {tuple(pixel_values.shape)}')
    outputs = videomae(pixel_values=pixel_values.to(DEVICE))
    features = outputs.last_hidden_state.mean(dim=1)
    return features.cpu()

@torch.inference_mode()
def extract_video_features(video_path: Path, batch_size: int=FEATURE_BATCH_SIZE):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'動画を開けません: {video_path}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    starts = list(range(0, max(total - CLIP_FRAMES + 1, 1), FEATURE_STRIDE_FRAMES))
    last_start = max(total - CLIP_FRAMES, 0)
    if not starts or starts[-1] < last_start:
        starts.append(last_start)
    feature_batches = []
    frame_batch = []
    try:
        for start in tqdm(starts, desc='VideoMAE feature extraction'):
            frame_batch.append(read_clip(cap, start, CLIP_FRAMES))
            if len(frame_batch) == batch_size:
                feature_batches.append(encode_clips(frame_batch))
                frame_batch = []
        if frame_batch:
            feature_batches.append(encode_clips(frame_batch))
    finally:
        cap.release()
    features = torch.cat(feature_batches, dim=0).numpy().astype(np.float32)
    return (features, np.asarray(starts, dtype=np.int64))

class LocalTransformerBlock(nn.Module):

    def __init__(self, dim=256, heads=8, window=19, dropout=0.1):
        super().__init__()
        self.window = window
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, x):
        T = x.size(1)
        idx = torch.arange(T, device=x.device)
        mask = (idx[:, None] - idx[None, :]).abs() > self.window
        y = self.norm1(x)
        x = x + self.attn(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))

class DetectionHead(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.tower = nn.Sequential(nn.Conv1d(dim, dim, 3, padding=1), nn.ReLU(), nn.Conv1d(dim, dim, 3, padding=1), nn.ReLU())
        self.cls = nn.Conv1d(dim, 1, 3, padding=1)
        self.reg = nn.Conv1d(dim, 2, 3, padding=1)
        nn.init.constant_(self.cls.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        h = self.tower(x)
        return (self.cls(h).squeeze(1), F.softplus(self.reg(h)).transpose(1, 2))

class CompactActionFormer(nn.Module):

    def __init__(self, input_dim=1024, dim=256, levels=4):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv1d(input_dim, dim, 3, padding=1), nn.ReLU())
        self.blocks = nn.ModuleList([LocalTransformerBlock(dim) for _ in range(3)])
        self.down = nn.ModuleList([nn.Conv1d(dim, dim, 3, stride=2, padding=1) for _ in range(levels - 1)])
        self.head = DetectionHead(dim)
        self.levels = levels

    def forward(self, x):
        h = self.proj(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        feats = [h.transpose(1, 2)]
        for layer in self.down:
            feats.append(F.relu(layer(feats[-1])))
        return [self.head(f) for f in feats]

@torch.inference_mode()
def predict_segments(model, features, score_threshold=0.05):
    model.eval()
    proposals = []
    window_size = MAX_SEQ_LEN
    window_stride = max(1, window_size // 2)
    for offset in range(0, len(features), window_stride):
        chunk = features[offset:offset + window_size]
        x = torch.from_numpy(chunk).unsqueeze(0).to(DEVICE)
        outputs = model(x)
        for (logits, regs), stride in zip(outputs, FPN_STRIDES):
            scores = logits[0].sigmoid().detach().cpu().numpy()
            reg = regs[0].detach().cpu().numpy()
            keep = scores >= score_threshold
            idx = np.flatnonzero(keep).astype(np.float32)
            centers = (idx + 0.5) * stride
            selected_reg = reg[keep] * stride
            starts = centers - selected_reg[:, 0]
            ends = centers + selected_reg[:, 1]
            for a, b, score in zip(starts, ends, scores[keep]):
                start = max(0.0, float(a) + offset)
                end = min(float(len(features)), float(b) + offset)
                if end > start:
                    proposals.append([start, end, float(score)])
        del outputs, x
    return np.asarray(proposals, dtype=np.float32).reshape(-1, 3)

def temporal_iou_matrix(one, many):
    inter = np.maximum(0, np.minimum(one[1], many[:, 1]) - np.maximum(one[0], many[:, 0]))
    union = np.maximum(many[:, 1], one[1]) - np.minimum(many[:, 0], one[0])
    return inter / np.maximum(union, 1e-08)

def soft_nms(proposals, sigma=0.5, min_score=0.1, max_segments=100):
    props = proposals.copy()
    selected = []
    while len(props) and len(selected) < max_segments:
        i = np.argmax(props[:, 2])
        best = props[i].copy()
        selected.append(best)
        props = np.delete(props, i, axis=0)
        if not len(props):
            break
        ious = temporal_iou_matrix(best, props)
        props[:, 2] *= np.exp(-ious ** 2 / sigma)
        props = props[props[:, 2] >= min_score]
    return np.asarray(selected)
