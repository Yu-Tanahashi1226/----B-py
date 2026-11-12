"""Count-first verification of the 24 reference-localized pitches; no graphic-loss requirement."""
import csv
import html
import json
from pathlib import Path
import subprocess

from common import read_jsonl,save_json,sha256
from event_count_gate import count_gate
from scoreboard_rule_report import run_validation
from run_foul_pipeline import resolve_config,preflight,environment,stage

ROOT=Path(__file__).resolve().parent


def analyze(job,matches,source,ffmpeg):
    from reference_worker import export_clip
    cfg=resolve_config(job['config']);check=preflight(cfg)
    if not check['ready']:raise ValueError(json.dumps(check,ensure_ascii=False))
    output=Path(job['output'])/'analysis';output.mkdir(exist_ok=True)
    accepted={m['reference']:m for m in matches if m['status']=='matched'}
    if not accepted:
        save_json(output/'events.json',[])
        (output/'index.html').write_text('<meta charset="utf-8"><p>映像を一意に照合できた参考場面がないため、BSO・VideoMAE確認は実行していません。</p>',encoding='utf-8')
        return
    frames=set();period=cfg['sample_period_sec'];fps=source['fps']
    ranges=[(0,source['duration_sec'])] if job['scoreboard_scope']=='full' else [(max(0,m['start_sec']-5),min(source['duration_sec'],m['end_sec']+10)) for m in accepted.values()]
    for a,b in ranges:
        frames.update(min(source['frames']-1,round(i*period*fps)) for i in range(max(0,int(a/period)),int(b/period)+1))
    board_out=output/'scoreboard';board_out.mkdir(exist_ok=True)
    board_job=dict(video=job['video'],output=str(board_out),sha256=source['source_sha256'],sample_frames=sorted(frames),config=cfg,
                   **{k:source[k] for k in ['frames','fps','duration_sec','width','height']})
    board_job['model_hashes']={name:sha256(cfg['yolo'][name]) for name in ['weights','inner_weights']}
    board_path=board_out/'job.json'
    if board_path.exists() and json.loads(board_path.read_text())!=board_job:raise ValueError('Scoreboard inputs/settings changed; use a new output folder')
    save_json(board_path,board_job);env=environment(output)
    stage(board_job,'yolo','scoreboard_worker.py',['yolo',str(board_path)],env,True)
    stage(board_job,'ocr','scoreboard_worker.py',['ocr',str(board_path)],env,True)
    raw_observations=read_jsonl(board_out/'observations.jsonl')
    observations=run_validation(board_job)
    print(f'野球ルール確認: {board_out / "rules" / "review.html"}',flush=True)
    from report import export_images
    save_json(board_out/'diagnostics.json',export_images(board_job,raw_observations,{'episodes':[]}))
    events=[]
    for ref in job['references']:
        match=accepted.get(ref['filename'])
        if not match:continue
        for event in ref['events']:
            events.append(dict(event_id=len(events)+1,reference=ref['filename'],reference_sha256=ref['sha256'],pitch_index=event['pitch_index'],
                               reference_offset_sec=match['offset_sec'],reference_duration_sec=ref['duration_sec'],
                               start_sec=match['offset_sec']+event['start_sec'],end_sec=match['offset_sec']+event['end_sec']))
    events.sort(key=lambda e:e['start_sec'])
    for i,event in enumerate(events):
        next_start=events[i+1]['start_sec'] if i+1<len(events) else None
        event['count_gate']=count_gate(observations,event['start_sec'],event['end_sec'],next_start,period)
    verify_job=dict(video=job['video'],source_sha256=source['source_sha256'],fps=fps,frames=source['frames'],duration_sec=source['duration_sec'],
                    output=str(output),config=cfg,events=events,head_sha256=sha256(ROOT/'reference_outcome_heads.json'))
    verify_path=output/'verification_job.json';save_json(verify_path,verify_job)
    with (output/'videomae.log').open('w',encoding='utf-8') as log:
        subprocess.run([cfg['python']['pitch'],str(ROOT/'matched_videomae.py'),str(verify_path)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    results=json.loads((output/'events.json').read_text())
    candidates=output/'foul_candidates';candidates.mkdir(exist_ok=True)
    for event in results:
        if event['status']=='model_supported_foul_candidate':
            name=Path(event['reference']).stem+f'__pitch{event["pitch_index"]:02d}_review.mp4'
            export_clip(job['video'],candidates/name,max(0,event['model_start_sec']-1),min(source['duration_sec'],event['model_end_sec']+4),ffmpeg,source['source_sha256'])
            event['candidate_clip']='foul_candidates/'+name
    save_json(output/'events.json',results)
    flat=[]
    for event in results:
        row={k:v for k,v in event.items() if k not in ['count_gate']}
        row['count_gate_reason']=event['count_gate']['reason']
        for phase in ['before','after']:
            row.update({phase+'_'+key:(event['count_gate'].get(phase) or {}).get(key) for key in ['balls','strikes','outs']})
        flat.append(row)
    fields=['event_id','reference','pitch_index','start_sec','end_sec','status','count_gate_reason','before_balls','before_strikes','before_outs','after_balls','after_strikes','after_outs',
            'model_start_sec','model_end_sec','pitch_model_score','foul_model_score','foul_model_decision','review_required','candidate_clip']
    with (output/'analysis.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(flat)
    # All scoreboard fields remain available even when an event cannot be judged.
    with (output/'scoreboard.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields2=['frame','time_sec','balls','strikes','outs','runner_first','runner_second','runner_third','inning','half','upper_team','lower_team','upper_score','lower_score','count_comparison_allowed','held_fields']
        writer=csv.DictWriter(stream,fieldnames=fields2);writer.writeheader()
        for row in observations:
            writer.writerow(dict(frame=row['frame'],time_sec=row['time_sec'],**{k:(row.get('count') or {}).get(k) for k in ['balls','strikes','outs']},
                                 count_comparison_allowed=row.get('rule_validation',{}).get('count_comparison_allowed'),
                                 held_fields=';'.join(k for k,v in row.get('rule_validation',{}).get('fields',{}).items() if v['status']=='held'),
                                 **{'runner_'+k:(row.get('runners') or {}).get(k) for k in ['first','second','third']},
                                 **{k:(row.get('scoreboard') or {}).get(k) for k in ['inning','half','upper_team','lower_team','upper_score','lower_score']}))
    text='<meta charset="utf-8"><style>body{font:17px/1.7 sans-serif;margin:30px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}video{max-width:750px;width:100%}</style><h1>BSOとVideoMAEの確認</h1><p>映像照合で投球時刻の候補を求め、2ストライク・BSO不変のものだけVideoMAEに渡しました。投球スコアと試作ファウル判別スコアを分けて表示します。どちらもファウルの確定判定ではありません。</p><p><a href="analysis.csv">投球別CSV</a> / <a href="scoreboard.csv">BSO・走者・文字情報CSV</a> / <a href="scoreboard/annotation_review/review.csv">注釈確認表</a></p><table><tr><th>参考場面</th><th>投球</th><th>判定</th><th>投球スコア</th><th>試作ファウルスコア</th></tr>'
    text=text.replace('<table><tr>', '<p>スコアボードCSVには未検出・不整合時に保持した直前値を含みます。<a href="scoreboard/rules/review.html">野球ルールによる補正前後・保留理由・評価表</a>で新しい観測と区別できます。保持値だけではBSO不変の条件を満たしません。</p><table><tr>',1)
    for event in results:
        text+='<tr>'+''.join('<td>'+html.escape(str(event.get(k,'')))+'</td>' for k in ['reference','pitch_index','status','pitch_model_score','foul_model_score'])+'</tr>'
    text+='</table>'
    for event in results:
        if event.get('candidate_clip'):text+=f'<h2>{html.escape(event["reference"])}・{event["pitch_index"]}球目（要確認）</h2><video controls preload="metadata" src="{event["candidate_clip"]}"></video>'
    (output/'index.html').write_text(text,encoding='utf-8')
