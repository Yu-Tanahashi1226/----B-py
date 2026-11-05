"""Export actual observations, gated results and blank annotation-review tasks."""
import argparse
import csv
import html
import json
from pathlib import Path
from collections import Counter

from common import frames_at, read_jsonl, save_json

REASONS = {
    'before_bso_unreadable': '消える前のBSOを安定して読めない',
    'after_bso_unreadable': '再表示後のBSOを比較できない（表示時間・読取りを確認）',
    'before_not_two_strikes': '消える前が2ストライクではない',
    'bso_changed_or_delayed_update': 'BSOが変化、または遅れて更新された',
    'context_changed': '走者・得点・回などが変化した',
    'gap_too_long': '表示のない時間が長く、対応を判断できない',
    'two_strikes_bso_unchanged': '再表示後も2ストライク・BSO不変',
}
STATUS = {
    'inferred_two_strike_foul': '2ストライク後のファウルと推定（要確認）',
    'pitch_not_supported': '投球の十分な検出なし（見逃しの可能性あり）',
    'comparison_pitch_detected': '直後の投球の比較用検出',
}


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def build_events(job, gate, proposals):
    output = []
    episodes = {e['episode_id']: e for e in gate['episodes']}
    for request in gate['requests']:
        episode = episodes[request['episode_id']]
        members = [p for p in proposals if int(p['window_id']) == request['window_id']]
        eligible = [p for p in members if float(p['start_sec']) < request['pitch_must_start_before_sec']]
        if request['role'] == 'following_pitch_comparison':
            strong = [p for p in eligible if float(p['model_score']) >= job['config']['pitch']['pitch_confirmation_score']]
            best = min(strong, key=lambda p: float(p['start_sec'])) if strong else None
        else:
            best = max(eligible, key=lambda p: float(p['model_score'])) if eligible else None
        supported = best is not None and float(best['model_score']) >= job['config']['pitch']['pitch_confirmation_score']
        status = ('inferred_two_strike_foul' if request['role'] == 'foul_candidate' else 'comparison_pitch_detected') if supported else 'pitch_not_supported'
        comparison = request['role'] == 'following_pitch_comparison'
        output.append(dict(**request, before=episode['after'] if comparison else episode['before'],
                           after=None if comparison else episode['after'],
                           status=status, pitch_supported_by_model=supported, confirmed_foul=None,
                           review_required=True, model_start_sec=float(best['start_sec']) if best else None,
                           model_end_sec=float(best['end_sec']) if best else None,
                           model_score=float(best['model_score']) if best else None,
                           proposal_count=len(members), eligible_proposal_count=len(eligible),
                           missing_context=episode.get('missing_context', [])))
    return output


def diagnostic_tasks(rows, maximum=40):
    """Select difficult and ordinary images; never infer GT from a missing box."""
    tasks = {}
    for index, row in enumerate(rows):
        reasons = []
        visible = row['board_layout_status'] == 'supported_corner_layout'
        if visible:
            a,b,c,d = row['board']['box']
            if c-a < 180 or d-b < 70:
                reasons.append('small_scoreboard_roi_check_native_resolution')
        if row.get('board') and not visible:
            reasons.append('wrong_or_unsupported_board_layout')
        if visible and row.get('count') is None:
            reasons.append('board_visible_bso_unreadable')
        if visible and any(x['confidence'] < .4 for x in row.get('inner_detections', [])):
            reasons.append('low_confidence_inner_detection')
        if index and index+1 < len(rows):
            previous, following = rows[index-1], rows[index+1]
            if previous.get('count') and previous.get('count') == following.get('count') and row.get('count') != previous['count']:
                reasons.append('bso_single_sample_disagreement')
            if previous.get('runners') is not None and previous.get('runners') == following.get('runners') and row.get('runners') != previous['runners']:
                reasons.append('runner_single_sample_disagreement')
            if visible and previous.get('board_layout_status') == 'supported_corner_layout' and row.get('count') == previous.get('count'):
                pre_context, context = previous.get('scoreboard') or {}, row.get('scoreboard') or {}
                if any(pre_context.get(k) is not None and context.get(k) is not None and pre_context[k] != context[k] for k in ['inning','half']):
                    reasons.append('ocr_inning_or_half_disagreement')
            if row.get('board') is None and (previous.get('board') or following.get('board')):
                reasons.append('board_presence_boundary_check_true_absence_or_miss')
        if visible and row.get('scoreboard') and any(row['scoreboard'].get(k) is None for k in ['upper_team','lower_team','upper_score','lower_score','inning','half']):
            reasons.append('ocr_missing_field')
        if reasons:
            tasks[row['frame']] = dict(frame=row['frame'], time_sec=row['time_sec'], reasons=reasons, row=row)
    # Spread each reason across the source video; avoid 40 adjacent copies.
    selected = {}
    all_reasons = sorted({reason for t in tasks.values() for reason in t['reasons']})
    budget = max(1, maximum // max(1, len(all_reasons)+1))
    for reason in all_reasons:
        choices = [t for t in tasks.values() if reason in t['reasons']]
        for index in range(min(budget, len(choices))):
            task = choices[round(index * (len(choices)-1) / max(1, min(budget,len(choices))-1))]
            selected[task['frame']] = task
    for index in range(min(maximum-len(selected), 6, len(rows))):
        row = rows[round(index * (len(rows)-1) / max(1, min(6,len(rows))-1))]
        selected.setdefault(row['frame'], dict(frame=row['frame'], time_sec=row['time_sec'], reasons=['ordinary_control_sample'], row=row))
    return sorted(selected.values(), key=lambda t: t['frame'])[:maximum], Counter(reason for t in tasks.values() for reason in t['reasons'])


def export_images(job, rows, gate):
    import cv2
    output = Path(job['output'])
    evidence = output / 'evidence'
    review = output / 'annotation_review'
    for directory in [evidence, review / 'images', review / 'inner_images', review / 'labels']:
        directory.mkdir(parents=True, exist_ok=True)
    selected, reason_counts = diagnostic_tasks(rows, job['config'].get('diagnostics', {}).get('maximum_images', 40))
    by_frame = {r['frame']: r for r in rows}
    evidence_frames = set()
    for episode in gate['episodes']:
        if episode['selected'] or episode['reason'] in ['before_bso_unreadable', 'after_bso_unreadable', 'bso_changed_or_delayed_update']:
            for sec in [episode['last_visible_sec'], episode['reappear_sec'], (episode.get('after') or {}).get('last_sec', episode['reappear_sec'])]:
                evidence_frames.add(min(job['frames']-1, round(sec*job['fps'])))
    wanted = sorted(evidence_frames | {t['frame'] for t in selected})
    task_lookup = {t['frame']: t for t in selected}
    for number, frame in frames_at(job['video'], wanted):
        row = by_frame.get(number)
        if row is None:
            continue
        if number in evidence_frames:
            cv2.imwrite(str(evidence / f'{number:08d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if number in task_lookup:
            task = task_lookup[number]
            name = f'{number:08d}.jpg'
            cv2.imwrite(str(review / 'images' / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            task['image'] = f'images/{name}'
            if row.get('inner_crop_box'):
                a,b,c,d = row['inner_crop_box']
                cv2.imwrite(str(review / 'inner_images' / name), frame[b:d,a:c], [cv2.IMWRITE_JPEG_QUALITY, 95])
                task['inner_image'] = f'inner_images/{name}'
            else:
                task['inner_image'] = ''
    flat = []
    for task in selected:
        row = task['row']
        flat.append(dict(video=Path(job['video']).name, source_sha256=job['sha256'], frame=task['frame'],
                         time_sec=task['time_sec'], reason=';'.join(task['reasons']), image=task.get('image',''),
                         inner_image=task.get('inner_image',''), predicted_board_present=row.get('board') is not None,
                         predicted_bso=json.dumps(row.get('count'), ensure_ascii=False),
                         predicted_runners=json.dumps(row.get('runners'), ensure_ascii=False),
                         true_board_present='', true_balls='', true_strikes='', true_outs='',
                         true_first='', true_second='', true_third='', note=''))
    fields = ['video','source_sha256','frame','time_sec','reason','image','inner_image','predicted_board_present','predicted_bso','predicted_runners',
              'true_board_present','true_balls','true_strikes','true_outs','true_first','true_second','true_third','note']
    write_csv(review / 'review.csv', flat, fields)
    save_json(review / 'model_predictions.json', [dict(frame=t['frame'], detections=t['row'].get('detections'),
              inner_crop_box=t['row'].get('inner_crop_box'), inner_detections=t['row'].get('inner_detections')) for t in selected])
    (review / 'classes_scoreboard.txt').write_text('scoreboard\n', encoding='utf-8')
    (review / 'classes_inner.txt').write_text('ball\nstrike\nout\nrunner\n', encoding='utf-8')
    (review / 'README.txt').write_text(
        'review.csvのtrue_*を目視で入力してください。予測は正解ではありません。\n'
        'imagesは元動画のフル画像：scoreboardをアノテーション。inner_imagesは余白付き切り抜き：点灯したball/strike/outと占有塁runnerをアノテーション。\n'
        '同じコマの全対象を漏れなく付けてください。空の塁や消灯ランプにrunner/ball等を付けないでください。\n'
        '現モデルの4クラスにはemptyがありません。消灯・空塁を明示分類する改善には別の設計と再学習が必要です。\n'
        'labelsは空です。未検出を正解の負例として自動保存しません。真に対象がない画像だけ、確認後に空ラベルにします。\n'
        '連続フレームをtrain/valにランダム分割せず、動画・試合単位で分けてください。\n'
        '学習データの枚数・多様性が未確認のため、この出力だけでアノテーション不足とは断定できません。\n', encoding='utf-8')
    return dict(selected_review_images=len(flat), reason_observations=dict(reason_counts),
                annotation_shortage_confirmed=False, review_csv='annotation_review/review.csv',
                limitation='Flags include real count changes, true graphic absence and OCR/layout errors. Human review is required before assigning a model error or training-data cause.')


def run(job):
    output = Path(job['output'])
    gate = json.loads((output / 'gate.json').read_text())
    rows = read_jsonl(output / 'observations.jsonl')
    with (output / 'pitch_proposals.csv').open(encoding='utf-8-sig', newline='') as stream:
        proposals = list(csv.DictReader(stream))
    events = build_events(job, gate, proposals)
    save_json(output / 'events.json', events)
    csv_events = []
    for event in events:
        flat = {k:v for k,v in event.items() if k not in ['before','after','missing_context']}
        flat['missing_context'] = ';'.join(event['missing_context'])
        for phase in ['before','after']:
            flat.update({phase+'_'+k:v for k,v in (event.get(phase) or {}).items()})
        csv_events.append(flat)
    event_fields = ['request_id','episode_id','parent_episode_id','role','window_id','start_frame','end_frame_exclusive',
                    'start_sec','end_sec','pitch_must_start_before_sec','rationale','status','pitch_supported_by_model',
                    'confirmed_foul','review_required','model_start_sec','model_end_sec','model_score','proposal_count',
                    'eligible_proposal_count','missing_context']
    state_fields = ['balls','strikes','outs','first_sec','last_sec','support','inning','half','upper_score','lower_score',
                    'runner_first','runner_second','runner_third']
    event_fields += [phase+'_'+key for phase in ['before','after'] for key in state_fields]
    write_csv(output / 'foul_candidates.csv', csv_events, event_fields)
    observation_rows = []
    for row in rows:
        observation_rows.append(dict(frame=row['frame'], time_sec=row['time_sec'], board_layout_status=row['board_layout_status'],
                                     **{k:(row.get('count') or {}).get(k) for k in ['balls','strikes','outs']},
                                     **{'runner_'+k:(row.get('runners') or {}).get(k) for k in ['first','second','third']},
                                     **{k:(row.get('scoreboard') or {}).get(k) for k in ['inning','half','upper_team','lower_team','upper_score','lower_score']},
                                     count_status=row['count_status'], ocr_status=row['ocr_status'], context_source_sec=row.get('context_source_sec')))
    write_csv(output / 'scoreboard.csv', observation_rows, list(observation_rows[0]) if observation_rows else ['frame','time_sec'])
    episodes = [{k:(json.dumps(v, ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in e.items()} for e in gate['episodes']]
    write_csv(output / 'gate_decisions.csv', episodes, list(dict.fromkeys(k for e in episodes for k in e)) or ['episode_id','reason','selected'])
    diagnostics = export_images(job, rows, gate)
    save_json(output / 'diagnostics.json', diagnostics)
    summary = dict(video=Path(job['video']).name, source_sha256=job['sha256'], duration_sec=job['duration_sec'],
                   scoreboard_observations=len(rows), disappearance_episodes=len(gate['episodes']), selected_episodes=gate['selected_episodes'],
                   videomae_windows=len(gate['windows']), inferred_fouls=sum(e['status']=='inferred_two_strike_foul' for e in events),
                   gate_reasons=dict(Counter(e['reason'] for e in gate['episodes'])), confirmed_fouls=None,
                   visibility_notes=gate.get('visibility_notes',[]),
                   whole_video_videomae_used=False, diagnostics=diagnostics,
                   interpretation='Rule + pitch-motion model inference, not confirmed foul ground truth. Count-unchanged events can include replays, pickoffs and graphic delays.')
    save_json(output / 'summary.json', summary)
    from plot_timeline import plot
    plot(job, rows, gate, proposals)
    make_html(job, rows, gate, events, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_html(job, rows, gate, events, summary):
    out = Path(job['output'])
    escape = lambda value: html.escape(str(value))
    details = []
    for episode in gate['episodes']:
        matching = [event for event in events if event['episode_id'] == episode['episode_id']]
        text = '<br>'.join(STATUS[event['status']] + (f' / 投球スコア {event["model_score"]:.3f}' if event['model_score'] is not None else '') for event in matching)
        images = []
        for sec in [episode['last_visible_sec'], episode['reappear_sec'], (episode.get('after') or {}).get('last_sec', episode['reappear_sec'])]:
            name = f'{min(job["frames"]-1,round(sec*job["fps"])):08d}.jpg'
            if (out / 'evidence' / name).exists():
                images.append(f'<figure><img loading="lazy" src="evidence/{name}"><figcaption>{sec:.2f}秒</figcaption></figure>')
        before = episode.get('before') or {}
        after = episode.get('after') or {}
        count_text = lambda value: '/'.join(str(value.get(k,'?')) for k in ['balls','strikes','outs'])
        details.append(f'<article><h3>表示消失 {episode["episode_id"]}: {episode["disappear_sec"]:.2f}–{episode["reappear_sec"]:.2f}秒</h3>'
                       f'<p>{REASONS[episode["reason"]]} / 前BSO {count_text(before)} → 後BSO {count_text(after)}</p>'
                       f'<p>{text}</p><button data-time="{max(0,episode["last_visible_sec"]-8):.3f}">動画のこの場面へ</button>'
                       '<div class="pictures">'+''.join(images)+'</div></article>')
    duration = job['duration_sec']
    lines = ['<svg viewBox="0 0 1100 240" role="img" aria-label="VideoMAE対象区間と投球候補">', '<text x="5" y="18">青：VideoMAEへ渡した範囲　赤：投球区間候補（縦軸＝投球スコア）</text>']
    for window in gate['windows']:
        x=50+window['start_sec']/duration*1000
        width=(window['end_sec']-window['start_sec'])/duration*1000
        lines.append(f'<rect x="{x:.2f}" y="40" width="{width:.2f}" height="150" fill="#b9dfe9"/>')
    with (out/'pitch_proposals.csv').open(encoding='utf-8-sig') as stream:
        for proposal in csv.DictReader(stream):
            x1=50+float(proposal['start_sec'])/duration*1000
            x2=50+float(proposal['end_sec'])/duration*1000
            y=190-float(proposal['model_score'])*150
            lines.append(f'<line x1="{x1:.2f}" x2="{x2:.2f}" y1="{y:.2f}" y2="{y:.2f}" stroke="red" stroke-width="3"/>')
    lines.extend(['<path d="M50 40V190H1050" fill="none" stroke="black"/>','<text x="20" y="45">1</text><text x="20" y="195">0</text>',f'<text x="50" y="218">0秒</text><text x="980" y="218">{duration:.1f}秒</text>','</svg>'])
    page = '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2ストライク後ファウル候補</title><style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:25px auto;padding:16px}article{border:1px solid #ccc;padding:14px;margin:20px 0}svg,video{width:100%}video{max-height:450px}.pictures{display:flex;gap:8px}figure{margin:0;flex:1;min-width:0}img{width:100%}a{margin-right:16px}button{padding:8px}</style>'
    page += f'<h1>{escape(Path(job["video"]).name)}</h1><p>スコアボード条件に一致：{summary["selected_episodes"]}場面。VideoMAE：{summary["videomae_windows"]}区間。ファウル推定：{summary["inferred_fouls"]}場面。</p>'
    if not summary['selected_episodes']:
        page += '<p>対象0件は、この表示条件で解析する場面を選べなかったことを示します。表示が消えない場合や、動画の終端で再表示・安定値を確認できない場合も含まれます。</p>'
    notes = {'short_absence_ignored':'1回だけの未検出を除外',
             'clip_starts_without_board_no_before_state':'動画の冒頭に表示がなく、消失前を比較できない',
             'clip_ends_without_board_no_reappearance':'動画が終わるまで再表示を確認できない',
             'absence_adjacent_to_unsupported_layout':'前後が別レイアウト等で比較できない'}
    if gate.get('visibility_notes'):
        page += '<ul>'+''.join(f'<li>{n["start_sec"]:.2f}〜{n["end_sec"]:.2f}秒：{notes[n["reason"]]}</li>' for n in gate['visibility_notes'])+'</ul>'
    page += '<p>スコアは投球らしさのモデル出力で、ファウル確率ではありません。表示の消失はYOLOの見落としの場合もあります。推定結果は動画で確認してください。</p>'
    page += '<p><a href="scoreboard.csv" download>全スコアボードCSV</a><a href="foul_candidates.csv" download>ファウル候補CSV</a><a href="gate_decisions.csv" download>対象・除外の理由</a><a href="annotation_review/review.csv" download>追加注釈の確認表</a></p>'
    page += '<p>青は正解区間ではなくVideoMAEの入力範囲です。走者の灰色は空塁という予測で、未検出との区別には確認が必要です。</p><img src="timeline.png" alt="対象区間・投球候補・BSO・走者・得点の時系列グラフ">'
    (out/'gate_timeline.svg').write_text(''.join(lines),encoding='utf-8')
    page += '<p>手元の元動画を選択すると、各場面へ移動できます（動画はアップロードされません）。</p><input id="file" type="file" accept="video/*"><video id="video" controls></video>'
    page += ''.join(details)
    page += '<script>const v=document.getElementById("video");let url;document.getElementById("file").addEventListener("change",e=>{if(url)URL.revokeObjectURL(url);url=URL.createObjectURL(e.target.files[0]);v.src=url});document.querySelectorAll("button[data-time]").forEach(b=>b.onclick=()=>{if(!v.src){alert("先に元動画を選んでください");return}v.currentTime=Number(b.dataset.time);v.scrollIntoView({behavior:"smooth"})});</script></html>'
    (out / 'index.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('job')
    args=parser.parse_args()
    run(json.loads(Path(args.job).read_text(encoding='utf-8-sig')))
