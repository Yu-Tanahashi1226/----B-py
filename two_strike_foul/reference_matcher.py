"""Retrieve known reference scenes using temporal image agreement, not foul labels."""
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
from common import metadata, save_json, sha256

DESCRIPTOR_VERSION = 'gray_grid_32x18_centered_v1'


def descriptor(frame):
    gray = cv2.cvtColor(cv2.resize(frame, (32,18), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    gray -= gray.mean()
    norm = float(np.linalg.norm(gray))
    return gray / max(norm, 1e-6)


def index_video(video, directory, period=.5):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    info=metadata(video)
    identity=dict(source_sha256=sha256(video),period_sec=period,descriptor=DESCRIPTOR_VERSION,**info)
    meta=directory/'index.json'
    if meta.exists() and json.loads(meta.read_text())==identity:
        vectors=np.load(directory/'vectors.npy',allow_pickle=False)
        times=np.load(directory/'times.npy',allow_pickle=False)
        if vectors.shape!=(len(times),576) or not np.isfinite(vectors).all():raise ValueError('Invalid image index')
        return dict(vectors=vectors,times=times,identity=identity)
    cap=cv2.VideoCapture(str(video));vectors=[];times=[]
    step=max(1,round(period*info['fps']));started=last=time.monotonic()
    try:
        for number in range(info['frames']):
            if not cap.grab():raise RuntimeError(f'Cannot decode {video}: frame {number}')
            if number%step:continue
            ok,frame=cap.retrieve()
            if not ok:raise RuntimeError(f'Cannot retrieve frame {number}')
            vectors.append(descriptor(frame));times.append(number/info['fps'])
            if time.monotonic()-last>10:
                print(f'映像索引: {number/info["fps"]:.0f}/{info["duration_sec"]:.0f}秒 ({time.monotonic()-started:.0f}s)',flush=True);last=time.monotonic()
    finally:cap.release()
    vectors=np.asarray(vectors,dtype=np.float32);times=np.asarray(times,dtype=np.float64)
    np.save(directory/'vectors.npy',vectors);np.save(directory/'times.npy',times);save_json(meta,identity)
    return dict(vectors=vectors,times=times,identity=identity)


def anchor_indices(index, maximum=17):
    count=len(index['times'])
    if count<4:raise ValueError('Reference is too short for temporal verification')
    return np.unique(np.linspace(1,count-2,min(maximum,count-2)).round().astype(int))


def rank_candidates(source, reference, maximum=8):
    """Require simultaneous agreement at many ordered reference timestamps."""
    anchors=anchor_indices(reference)
    step=float(np.median(np.diff(source['times'])))
    offsets=np.rint(reference['times'][anchors]/step).astype(int)
    valid=len(source['times'])-int(offsets.max())
    if valid<=0:return []
    scores=np.zeros(valid,dtype=np.float32)
    for anchor,offset in zip(anchors,offsets):
        scores+=np.einsum('ij,j->i',source['vectors'][offset:offset+valid],reference['vectors'][anchor],optimize=True)
    scores/=len(anchors)
    remaining=scores.copy();selected=[]
    separation=max(2.,reference['identity']['duration_sec']*.4)
    radius=max(1,round(separation/step))
    for _ in range(min(maximum,len(scores))):
        position=int(remaining.argmax());score=float(remaining[position])
        if not np.isfinite(score):break
        selected.append(dict(offset_sec=float(source['times'][position]),coarse_score=score))
        remaining[max(0,position-radius):min(len(scores),position+radius+1)]=-np.inf
    return selected


def sample_descriptors(video, seconds, size=(96,54)):
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    numbers=[max(0,round(sec*fps)) for sec in seconds];images={};cursor=None
    try:
        for number in sorted(set(numbers)):
            if cursor is None or number-cursor>fps*2:
                cap.set(cv2.CAP_PROP_POS_FRAMES,number);cursor=number
            while cursor<=number:
                if not cap.grab():raise RuntimeError(f'Cannot read {video} at frame {cursor}')
                cursor+=1
            ok,frame=cap.retrieve()
            if not ok:raise RuntimeError(f'Cannot retrieve {video} at frame {number}')
            image=cv2.cvtColor(cv2.resize(frame,size,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
            image-=image.mean();image/=max(float(np.linalg.norm(image)),1e-6);images[number]=image
    finally:cap.release()
    return np.asarray([images[number] for number in numbers])


def refine_candidates(source_video, reference_video, candidates, source, reference):
    times=reference['times'][anchor_indices(reference,maximum=11)]
    ref=sample_descriptors(reference_video,times)
    duration=source['identity']['duration_sec'];ref_duration=reference['identity']['duration_sec']
    # Fine search uses native source frames, independently of the coarse index.
    refined=[]
    for candidate in candidates[:3]:
        best=None
        offsets=[candidate['offset_sec']+float(delta) for delta in np.linspace(-.5,.5,11)]
        offsets=[offset for offset in offsets if 0<=offset and offset+ref_duration<=duration+1/source['identity']['fps']]
        if not offsets:continue
        actuals=sample_descriptors(source_video,[float(t+offset) for offset in offsets for t in times]).reshape(len(offsets),len(times),-1)
        for offset,actual in zip(offsets,actuals):
            similarities=np.einsum('ij,ij->i',ref,actual)
            value=float(similarities.mean())
            if best is None or value>best['fine_score']:
                best=dict(candidate,offset_sec=offset,fine_score=value,
                          anchor_scores=[float(s) for s in similarities],anchor_times_sec=[float(t) for t in times])
        if best:refined.append(best)
    refined.sort(key=lambda c:c['fine_score'],reverse=True)
    if refined:
        best=refined[0];fps=source['identity']['fps']
        offsets=[best['offset_sec']+i/fps for i in range(-3,4)]
        offsets=[offset for offset in offsets if 0<=offset and offset+ref_duration<=duration+1/fps]
        frames=sample_descriptors(source_video,[float(t+offset) for offset in offsets for t in times]).reshape(len(offsets),len(times),-1)
        for offset,actual in zip(offsets,frames):
            similarities=np.einsum('ij,ij->i',ref,actual);score=float(similarities.mean())
            if score>best['fine_score']:
                best=dict(best,offset_sec=offset,fine_score=score,anchor_scores=[float(x) for x in similarities])
        refined[0]=best
    return refined


def locate(source_video, reference_video, source, reference, minimum=.90, margin=.025):
    candidates=rank_candidates(source,reference)
    candidates=refine_candidates(source_video,reference_video,candidates,source,reference)
    if not candidates:return dict(status='not_found',candidates=[])
    best=candidates[0]
    alternate=candidates[1]['fine_score'] if len(candidates)>1 else -1.
    # Mean agreement alone is insufficient: an ordered majority of independent
    # anchors must match, and a duplicated/replayed scene must remain ambiguous.
    anchor_fraction=sum(s>=minimum-.05 for s in best['anchor_scores'])/len(best['anchor_scores'])
    status='matched' if best['fine_score']>=minimum and anchor_fraction>=.8 else 'not_found'
    if status=='matched' and best['fine_score']-alternate<margin:status='ambiguous'
    return dict(status=status,offset_sec=best['offset_sec'],start_sec=best['offset_sec'],
                end_sec=min(source['identity']['duration_sec'],best['offset_sec']+reference['identity']['duration_sec']),
                similarity=best['fine_score'],alternative_similarity=alternate,
                anchor_fraction=anchor_fraction,candidates=candidates)
