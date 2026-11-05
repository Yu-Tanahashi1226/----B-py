"""VideoMAE only for count-unchanged pitches inside visually localized references."""
import argparse
import csv
import json
import math
from pathlib import Path

from common import save_json,sha256


def verify(job):
    output=Path(job['output']);cfg=job['config']['pitch']
    rows=[];ranges=[];models_ready=False
    heads_path=Path(__file__).with_name('reference_outcome_heads.json')
    if sha256(heads_path)!=job['head_sha256']:raise ValueError('Outcome heads changed during the run')
    heads=json.loads(heads_path.read_text(encoding='utf-8'))
    if heads['encoder']!=cfg['encoder'] or heads['revision']!=cfg['revision']:raise ValueError('Outcome head/encoder mismatch')
    for original in job['events']:
        row=dict(original,review_required=True,confirmed_foul=None,pitch_model_score=None,foul_model_score=None)
        if not row['count_gate']['selected']:
            row['status']=row['count_gate']['reason'];rows.append(row);continue
        if row['count_gate']['reason']!='two_strikes_bso_unchanged':raise ValueError('Invalid count gate')
        if not models_ready:
            import cv2
            import numpy as np
            import torch
            from transformers import AutoImageProcessor,VideoMAEModel
            import pitch_backend as backend
            from outcome_features import pool_context,predict_outcome
            torch.set_num_threads(cfg.get('threads',4));backend.DEVICE=torch.device(cfg.get('device','cpu'))
            detector=backend.CompactActionFormer(input_dim=768).eval().to(backend.DEVICE)
            detector.load_state_dict(torch.load(cfg['checkpoint'],map_location=backend.DEVICE,weights_only=True)['model'],strict=True)
            models_ready=True
        begin=max(row['reference_offset_sec'],row['start_sec']-.5)
        end=min(job['duration_sec'],row['reference_offset_sec']+row['reference_duration_sec'],row['end_sec']+4.)
        fps=job['fps'];origin=round(row['reference_offset_sec']*fps)
        first=origin+max(0,math.ceil((begin*fps-origin)/8))*8
        finish=min(job['frames'],math.ceil(end*fps))
        starts=list(range(first,finish-16+1,8))
        if not starts:
            row['status']='insufficient_native_video_context';rows.append(row);continue
        precision=cfg.get('encoder_precision','float32')
        if precision not in ['float32','bfloat16']:raise ValueError('Unsupported encoder precision')
        identity=dict(source_sha256=job['source_sha256'],reference_sha256=row['reference_sha256'],start_frame=first,end_frame_exclusive=finish,
                      encoder=cfg['encoder'],revision=cfg['revision'],precision=precision,clip_frames=16,stride_frames=8)
        directory=output/'selected_features'/str(row['event_id']);directory.mkdir(parents=True,exist_ok=True)
        meta=directory/'meta.json'
        reused=meta.exists() and json.loads(meta.read_text())==identity
        if reused:
            features=np.load(directory/'features.npy',allow_pickle=False)
            if not np.array_equal(np.load(directory/'starts.npy',allow_pickle=False),starts):raise ValueError('Selected feature starts changed')
        else:
            if not hasattr(backend,'videomae'):
                backend.processor=AutoImageProcessor.from_pretrained(cfg['encoder'],revision=cfg['revision'],use_fast=False)
                backend.videomae=VideoMAEModel.from_pretrained(cfg['encoder'],revision=cfg['revision']).eval().to(backend.DEVICE)
            from collections import deque
            cap=cv2.VideoCapture(job['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,first)
            ring=deque(maxlen=16);pending=[];batches=[];index=0
            try:
                for number in range(first,starts[-1]+16):
                    ok,frame=cap.read()
                    if not ok:raise RuntimeError(f'Cannot decode selected frame {number}')
                    ring.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    if index<len(starts) and number==starts[index]+15:
                        pending.append(list(ring));index+=1
                        if len(pending)==cfg.get('batch_size',2):
                            with torch.autocast(backend.DEVICE.type,dtype=torch.bfloat16,enabled=precision=='bfloat16'):
                                batches.append(backend.encode_clips(pending).float())
                            pending=[];print(f'event {row["event_id"]}: {index}/{len(starts)} clips',flush=True)
                if pending:
                    with torch.autocast(backend.DEVICE.type,dtype=torch.bfloat16,enabled=precision=='bfloat16'):
                        batches.append(backend.encode_clips(pending).float())
            finally:cap.release()
            features=torch.cat(batches).numpy().astype(np.float32)
            np.save(directory/'features.npy',features);np.save(directory/'starts.npy',np.asarray(starts,dtype=np.int64));save_json(meta,identity)
        if features.shape!=(len(starts),768) or not np.isfinite(features).all():raise ValueError('Invalid selected VideoMAE features')
        ranges.extend(dict(event_id=row['event_id'],reference=row['reference'],start_frame=s,end_frame_exclusive=s+16,cache_reused=reused) for s in starts)
        raw=backend.predict_segments(detector,features)
        kept=backend.soft_nms(raw,min_score=cfg['minimum_proposal_score']).reshape(-1,3)
        proposals=[]
        for a,b,score in kept:
            s=max(begin,(first+float(a)*8)/fps);e=min(end,(first+float(b)*8)/fps)
            if e>s:
                proposals.append(dict(start_sec=s,end_sec=e,score=float(score)))
        save_json(directory/'pitch_proposals.json',proposals)
        eligible=[p for p in proposals if min(p['end_sec'],row['end_sec'])>max(p['start_sec'],row['start_sec'])]
        best=max(eligible,key=lambda p:p['score']) if eligible else None
        row['pitch_model_score']=best['score'] if best else None
        row['model_start_sec']=best['start_sec'] if best else None
        row['model_end_sec']=best['end_sec'] if best else None
        vector=pool_context(features,(np.asarray(starts)+8)/fps,begin,end)
        row.update(predict_outcome(heads,row['reference'],vector))
        row['status']=('model_supported_foul_candidate' if row['foul_model_decision'] else 'model_did_not_support_foul_review') if best and best['score']>=cfg['pitch_confirmation_score'] else 'pitch_not_supported'
        rows.append(row)
    save_json(output/'events.json',rows)
    selected={e['event_id'] for e in rows if e['count_gate']['selected']}
    assert all(r['event_id'] in selected for r in ranges)
    with (output/'encoded_ranges.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['event_id','reference','start_frame','end_frame_exclusive','cache_reused']);writer.writeheader();writer.writerows(ranges)
    save_json(output/'verification_report.json',dict(events=len(rows),selected_by_count=len(selected),encoded_clips=len(ranges),
              outside_selected_events=0,blanket_whole_video_encoder=False,
              classifier_excludes_target_reference=True,confirmed_fouls=None,
              warning='Only 24 approximate training labels. Prototype outcome classification requires visual review.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job');args=parser.parse_args()
    verify(json.loads(Path(args.job).read_text(encoding='utf-8')))
