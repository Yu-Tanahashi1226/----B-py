"""Choose VideoMAE inputs BEFORE pitch inference, solely from board observations.

Times are source-video seconds. Missing board != a zero count. A supported board
whose lamps could not be read is UNKNOWN, not ABSENT. These rules are hypotheses,
not a trained foul classifier.
"""
COUNT_KEYS = ('balls', 'strikes', 'outs')
CONTEXT_KEYS = ('inning', 'half', 'upper_score', 'lower_score',
                'runner_first', 'runner_second', 'runner_third')
DEFAULTS = dict(min_absent_samples=2, min_visible_samples=2, stable_samples=3,
                pre_window_sec=5.0, post_settle_sec=1.0, post_window_sec=6.0,
                pitch_lookback_sec=8.0, post_padding_sec=0.5,
                max_gap_sec=45.0, max_sample_gap_factor=1.6,
                include_following_pitch=False, following_search_sec=45.0)


def presence(row):
    if row.get('board') is None:
        return 'absent'
    if row.get('board_layout_status') in ('unsupported_layout', 'ambiguous_layout'):
        return 'unknown'
    return 'visible'


def signature(row):
    if row.get('rule_validation', {}).get('count_comparison_allowed') is False:
        return None
    value = row.get('count')
    if not value or presence(row) != 'visible':
        return None
    maximums = (3, 2, 2)
    if not all(type(value.get(key)) is int and 0 <= value[key] <= maximum
               for key, maximum in zip(COUNT_KEYS, maximums)):
        return None
    return tuple(value[key] for key in COUNT_KEYS)


def stable_runs(rows, begin, end, period, samples=3, latest=False):
    runs, current = [], []
    for row in rows:
        if not begin <= row['time_sec'] <= end:
            continue
        sig = signature(row)
        if sig is None:
            if current:
                runs.append(current)
            current = []
            continue
        if current and (signature(current[-1]) != sig or row['time_sec'] - current[-1]['time_sec'] > period * 1.6):
            runs.append(current)
            current = []
        current.append(row)
    if current:
        runs.append(current)
    return [run for run in runs if len(run) >= samples]


def summarize(run):
    result = dict(zip(COUNT_KEYS, signature(run[0])))
    result.update(first_sec=run[0]['time_sec'], last_sec=run[-1]['time_sec'], support=len(run))
    for key in CONTEXT_KEYS:
        trusted = [r for r in run if r.get('rule_validation', {}).get('fields', {}).get(key, {}).get('status', 'observed') in ('observed', 'cached')]
        if key.startswith('runner_'):
            values = [(r.get('runners') or {}).get(key[7:]) for r in trusted]
        else:
            values = [(r.get('scoreboard') or {}).get(key) for r in trusted]
        known = [value for value in values if value is not None]
        result[key] = known[0] if known and all(v == known[0] for v in known) else None
    return result


def disappearance_episodes(rows, period, cfg):
    """Debounce consecutive ABSENT samples; UNKNOWN cannot establish a gap."""
    rows = sorted(rows, key=lambda r: r['time_sec'])
    episodes = []
    i = 0
    last_visible_start = None
    while i < len(rows):
        if presence(rows[i]) != 'visible':
            i += 1
            continue
        visible_start = i
        while i + 1 < len(rows) and presence(rows[i + 1]) == 'visible' and rows[i + 1]['time_sec'] - rows[i]['time_sec'] <= period * cfg['max_sample_gap_factor']:
            i += 1
        last_visible = i
        last_visible_start = visible_start
        j = i + 1
        if j >= len(rows) or presence(rows[j]) != 'absent':
            i += 1
            continue
        gap_start = j
        while j < len(rows) and presence(rows[j]) == 'absent' and (j == gap_start or rows[j]['time_sec'] - rows[j - 1]['time_sec'] <= period * cfg['max_sample_gap_factor']):
            j += 1
        absent_count = j - gap_start
        if j < len(rows) and presence(rows[j]) == 'visible':
            returned = j
            while j + 1 < len(rows) and presence(rows[j + 1]) == 'visible' and rows[j + 1]['time_sec'] - rows[j]['time_sec'] <= period * cfg['max_sample_gap_factor']:
                j += 1
            if (absent_count >= cfg['min_absent_samples'] and j - returned + 1 >= cfg['min_visible_samples']
                    and rows[gap_start]['time_sec'] - rows[last_visible]['time_sec'] <= period * cfg['max_sample_gap_factor']
                    and rows[returned]['time_sec'] - rows[returned - 1]['time_sec'] <= period * cfg['max_sample_gap_factor']):
                episodes.append(dict(last_visible_sec=rows[last_visible]['time_sec'],
                                     visible_block_start_sec=rows[last_visible_start]['time_sec'],
                                     disappear_sec=rows[gap_start]['time_sec'],
                                     reappear_sec=rows[returned]['time_sec'],
                                     post_visible_end_sec=rows[j]['time_sec'], absent_samples=absent_count))
        i = gap_start  # Visit returned block again; it may precede the next gap.
    return episodes


def make_gate(rows, fps, total_frames, period=0.5, settings=None):
    cfg = dict(DEFAULTS, **(settings or {}))
    rows = sorted(rows, key=lambda r: r['time_sec'])
    if any(b['time_sec'] <= a['time_sec'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Observation timestamps must be unique and increasing')
    episodes = disappearance_episodes(rows, period, cfg)
    visibility_notes = []
    index = 0
    while index < len(rows):
        if presence(rows[index]) != 'absent':
            index += 1
            continue
        begin = index
        while index+1 < len(rows) and presence(rows[index+1]) == 'absent':
            index += 1
        reason = None
        if index-begin+1 < cfg['min_absent_samples']:
            reason = 'short_absence_ignored'
        elif begin == 0:
            reason = 'clip_starts_without_board_no_before_state'
        elif index == len(rows)-1:
            reason = 'clip_ends_without_board_no_reappearance'
        elif presence(rows[begin-1]) != 'visible' or presence(rows[index+1]) != 'visible':
            reason = 'absence_adjacent_to_unsupported_layout'
        if reason:
            visibility_notes.append(dict(start_sec=rows[begin]['time_sec'],end_sec=rows[index]['time_sec'],reason=reason))
        index += 1
    for index, episode in enumerate(episodes):
        episode.update(episode_id=index + 1, selected=False, before=None, after=None, missing_context=[])
        begin = max(episode['visible_block_start_sec'], episode['last_visible_sec'] - cfg['pre_window_sec'])
        pre_runs = stable_runs(rows, begin, episode['last_visible_sec'], period, cfg['stable_samples'])
        post_begin = episode['reappear_sec'] + cfg['post_settle_sec']
        post_end = min(episode['post_visible_end_sec'], episode['reappear_sec'] + cfg['post_window_sec'])
        post_runs = stable_runs(rows, post_begin, post_end, period, cfg['stable_samples'])
        before = summarize(pre_runs[-1]) if pre_runs else None
        after = summarize(post_runs[-1]) if post_runs else None
        episode.update(before=before, after=after, post_check_begin_sec=post_begin, post_check_end_sec=post_end,
                       post_stable_states=[summarize(run) for run in post_runs])
        post_rows = [row for row in rows if post_begin <= row['time_sec'] <= post_end]
        episode['post_available_samples'] = len(post_rows)
        episode['required_stable_samples'] = cfg['stable_samples']
        if after is None:
            episode['after_failure_detail'] = ('visible_period_too_short_after_settle' if len(post_rows) < cfg['stable_samples']
                                               else 'bso_unreadable_or_not_consecutive_and_stable')
        if episode['reappear_sec'] - episode['disappear_sec'] > cfg['max_gap_sec']:
            reason = 'gap_too_long'
        elif before is None or episode['last_visible_sec'] - before['last_sec'] > period * 1.6:
            reason = 'before_bso_unreadable'
        elif before['strikes'] != 2:
            reason = 'before_not_two_strikes'
        elif after is None:
            reason = 'after_bso_unreadable'
        elif any(tuple(summarize(run)[k] for k in COUNT_KEYS) != tuple(before[k] for k in COUNT_KEYS) for run in post_runs):
            reason = 'bso_changed_or_delayed_update'
        else:
            changes = [k for k in CONTEXT_KEYS if before.get(k) is not None and after.get(k) is not None and before[k] != after[k]]
            episode['context_changes'] = changes
            episode['missing_context'] = [k for k in CONTEXT_KEYS if before.get(k) is None or after.get(k) is None]
            reason = 'context_changed' if changes else 'two_strikes_bso_unchanged'
        episode['reason'] = reason
        episode['selected'] = reason == 'two_strikes_bso_unchanged'
    requests = []
    for index, episode in enumerate(episodes):
        if not episode['selected']:
            continue
        targets = [(episode, 'foul_candidate', episode['episode_id'])]
        for target, role, parent in targets:
            # The pitch often ends just BEFORE the graphic disappears. Include
            # pre-disappearance video; never search only the absent-board interval.
            begin = max(target['visible_block_start_sec'], target['last_visible_sec'] - cfg['pitch_lookback_sec'])
            end = min(total_frames / fps, target['reappear_sec'] + cfg['post_padding_sec'])
            start_frame, end_frame = max(0, int(begin * fps)), min(total_frames, int(end * fps + 0.999999))
            if end_frame - start_frame < 16:
                continue
            requests.append(dict(request_id=len(requests) + 1, episode_id=target['episode_id'], parent_episode_id=parent,
                                 role=role, start_frame=start_frame, end_frame_exclusive=end_frame,
                                 start_sec=start_frame / fps, end_sec=end_frame / fps,
                                 pitch_must_start_before_sec=target['disappear_sec'],
                                 rationale='scoreboard gate; no whole-video pitch inference'))
        if cfg['include_following_pitch']:
            # The next pitch may keep the scoreboard on-screen, or the supplied
            # clip may end before it reappears. Do not misidentify the next
            # disappearance as necessarily the next pitch. Search a bounded,
            # explicitly enabled comparison window after the qualifying scene.
            begin = episode['reappear_sec'] + cfg['post_padding_sec']
            end = min(total_frames / fps, episode['reappear_sec'] + cfg['following_search_sec'])
            start_frame = max(0, int(begin * fps))
            end_frame = min(total_frames, int(end * fps + .999999))
            if end_frame-start_frame >= 16:
                requests.append(dict(request_id=len(requests)+1,episode_id=episode['episode_id'],
                                     parent_episode_id=episode['episode_id'],role='following_pitch_comparison',
                                     start_frame=start_frame,end_frame_exclusive=end_frame,
                                     start_sec=start_frame/fps,end_sec=end_frame/fps,
                                     pitch_must_start_before_sec=end_frame/fps,
                                     rationale='Explicitly enabled, bounded next-pitch comparison; not a foul-rule match.'))
    # Each unique source-frame range is encoded once. Multiple requests can link
    # to it; distant scenes are NEVER concatenated into one temporal sequence.
    windows = []
    for request in requests:
        key = (request['start_frame'], request['end_frame_exclusive'])
        window = next((w for w in windows if (w['start_frame'], w['end_frame_exclusive']) == key), None)
        if window is None:
            window = dict(window_id=len(windows) + 1, start_frame=key[0], end_frame_exclusive=key[1],
                          start_sec=key[0] / fps, end_sec=key[1] / fps, request_ids=[])
            windows.append(window)
        window['request_ids'].append(request['request_id'])
        request['window_id'] = window['window_id']
    return dict(schema_version=1, scope='scoreboard_gated_only', settings=cfg,
                sample_period_sec=period, episodes=episodes, requests=requests, windows=windows,
                selected_episodes=sum(e['selected'] for e in episodes), visibility_notes=visibility_notes,
                interpretation='Same BSO after disappearance is a candidate, not proof of a foul.')
