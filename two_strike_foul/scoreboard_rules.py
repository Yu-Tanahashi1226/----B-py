"""Causal baseball-state validation, separate from immutable detector/OCR output.

Held values are estimates for the record, never new evidence of an unchanged
count. Rules reject inconsistent readings, not prove that a model made an error.
"""
from copy import deepcopy
import math
import unicodedata


COUNT_FIELDS = ('balls', 'strikes', 'outs')
RUNNER_FIELDS = ('runner_first', 'runner_second', 'runner_third')
BOARD_FIELDS = ('inning', 'half', 'upper_team', 'lower_team', 'upper_score', 'lower_score')
FIELDS = COUNT_FIELDS + RUNNER_FIELDS + BOARD_FIELDS
DEFAULTS = dict(enabled=True, min_stable_samples=2, ocr_stable_samples=2,
                confirmation_window_sec=6.0, gap_review_sec=15.0,
                reset_evidence_max_age_sec=8.0,
                initial_state={}, team_aliases={})

REASONS = {
    'observed': 'ルールと連続確認を満たした読取り',
    'cached_context': '同じ表示区間のOCRキャッシュ',
    'waiting_for_confirmation': '連続した読取りによる確認待ち',
    'missing_or_unreadable': '未検出・読取り不能のため直前値を保持',
    'no_supported_scoreboard': 'スコアボードが見えないため直前値を保持',
    'invalid_value': '値が取り得る範囲・形式に合わない',
    'team_name_changed': '同じ試合のチーム名と一致しない',
    'score_decreased': '同じ試合で得点が減少した',
    'outs_decreased_without_inning_change': '回の交代を確認できないアウト数の減少',
    'count_decreased_without_batter_change_evidence': '打者交代の根拠を確認できないB/Sの減少',
    'inning_moved_backwards': '回・表裏が過去の状態に戻った',
    'inning_jump_needs_review': '観測が続く中で回・表裏が大きく飛んだ',
    'partial_inning_change': '回・表裏の片方だけでは交代を確認できない',
    'runner_drop_without_out_or_run_evidence': 'アウト・得点・回交代を確認できない走者数の減少',
    'inning_change': '回・表裏の交代に伴う正当なリセット',
    'possible_batter_change': 'アウト・出塁・得点の変化を伴うカウントリセット',
    'resynchronized_after_gap': '観測間隔が空いた後の回・表裏の再確認',
}


def flatten(row):
    count, runners, board = row.get('count') or {}, row.get('runners') or {}, row.get('scoreboard') or {}
    return dict(**{k: count.get(k) for k in COUNT_FIELDS},
                **{'runner_' + k: runners.get(k) for k in ('first', 'second', 'third')},
                **{k: board.get(k) for k in BOARD_FIELDS})


def team_key(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', str(value)).casefold()
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def normalize(field, value):
    if value is None or value == '':
        return None
    if field in ('upper_team', 'lower_team'):
        return unicodedata.normalize('NFKC', str(value)).strip() or None
    if field == 'half':
        return {'top': 'top', 'bottom': 'bottom', '表': 'top', '裏': 'bottom',
                'ウラ': 'bottom', 'オモテ': 'top'}.get(str(value).strip())
    if field in RUNNER_FIELDS:
        return value if type(value) is bool else None
    if type(value) is not int:
        return None
    maximum = {'balls': 3, 'strikes': 2, 'outs': 2}.get(field)
    if value < (1 if field == 'inning' else 0) or (maximum is not None and value > maximum):
        return None
    return value


def phase(state):
    inning, half = state.get('inning'), state.get('half')
    return 2 * (inning - 1) + (half == 'bottom') if inning is not None and half in ('top', 'bottom') else None


def visible(row):
    return row.get('board') is not None and row.get('board_layout_status') not in ('unsupported_layout', 'ambiguous_layout')


class RuleFilter:
    def __init__(self, settings=None):
        self.cfg = dict(DEFAULTS, **(settings or {}))
        for key in ('min_stable_samples', 'ocr_stable_samples'):
            if type(self.cfg[key]) is not int or self.cfg[key] < 1:
                raise ValueError(key + ' must be a positive integer')
        for key in ('confirmation_window_sec', 'gap_review_sec', 'reset_evidence_max_age_sec'):
            if not math.isfinite(self.cfg[key]) or self.cfg[key] <= 0:
                raise ValueError(key + ' must be positive and finite')
        self.values = dict.fromkeys(FIELDS)
        for key, value in self.cfg['initial_state'].items():
            if key not in FIELDS or normalize(key, value) is None:
                raise ValueError('Invalid configured initial_state: ' + key)
            self.values[key] = normalize(key, value)
        self.aliases = {team_key(k): str(v) for k, v in self.cfg['team_aliases'].items()}
        self.pending, self.observed_at = {}, {}
        self.serial = self.count_serial = self.outs_serial = self.runner_serial = 0
        self.reset_support = {}
        self.previous_time = None
        self.resync_phase = None
        self.audit = []

    def stable(self, key, value, now, fresh, samples):
        if not fresh:
            return False
        old = self.pending.get(key)
        n = old[1] + 1 if old and old[0] == value and now - old[2] <= self.cfg['confirmation_window_sec'] else 1
        self.pending[key] = (value, n, now)
        return n >= samples

    def apply(self, row):
        now = float(row['time_sec'])
        if not math.isfinite(now) or (self.previous_time is not None and now <= self.previous_time):
            raise ValueError('Observation times must be finite, unique and increasing')
        gap = now - self.previous_time if self.previous_time is not None else 0
        self.previous_time = now
        result = deepcopy(row)
        if not self.cfg['enabled']:
            return result
        raw = flatten(row)
        self.reset_support = {k: t for k, t in self.reset_support.items()
                              if now - t <= self.cfg['reset_evidence_max_age_sec']}
        vals = {key: normalize(key, raw[key]) for key in FIELDS}
        for key in ('upper_team', 'lower_team'):
            if vals[key] is not None:
                vals[key] = self.aliases.get(team_key(vals[key]), vals[key])
        on_screen = visible(row)
        fresh_ocr = row.get('ocr_status') not in ('same_visible_block_cached', 'not_scheduled', 'no_supported_scoreboard')
        # A contradictory observation breaks confirmation. Cached OCR can neither
        # confirm a new value nor count as an independent second reading.
        pending_values = dict(bs_pair=(vals['balls'], vals['strikes']), outs=vals['outs'],
                              runners=tuple(vals[k] for k in RUNNER_FIELDS))
        if fresh_ocr:
            pending_values.update({k: vals[k] for k in BOARD_FIELDS})
            pending_values['phase'] = phase(vals)
        for key, value in pending_values.items():
            if key in self.pending and (not on_screen or self.pending[key][0] != value):
                self.pending.pop(key)
        if not on_screen:
            self.pending.clear()
        info = {key: dict(status='held' if self.values[key] is not None else 'unknown',
                         reason='missing_or_unreadable' if on_screen else 'no_supported_scoreboard') for key in FIELDS}
        blocked = False

        def hold(keys, reason, blocking=False):
            nonlocal blocked
            blocked |= blocking
            for key in keys:
                info[key] = dict(status='held' if self.values[key] is not None else 'unknown', reason=reason)

        def accept(keys, reason='observed'):
            for key in keys:
                self.values[key] = vals[key]
                info[key] = dict(status='observed' if fresh_ocr or key not in BOARD_FIELDS else 'cached', reason=reason)
                if key not in BOARD_FIELDS or fresh_ocr:
                    self.observed_at[key] = now

        if on_screen:
            # A complete inning+half pair establishes a boundary. A partial OCR
            # change cannot authorize a drop of outs or reset the batting count.
            old_phase, new_phase = phase(self.values), phase(vals)
            if self.resync_phase != new_phase:
                self.resync_phase = None
            if gap > self.cfg['gap_review_sec']:
                self.resync_phase = new_phase
            if new_phase is not None:
                reason = None
                if old_phase is not None and new_phase < old_phase:
                    reason = 'inning_moved_backwards'
                elif old_phase is not None and new_phase > old_phase + 1 and self.resync_phase != new_phase:
                    reason = 'inning_jump_needs_review'
                if reason:
                    hold(('inning', 'half'), reason, True)
                elif new_phase == old_phase:
                    accept(('inning', 'half'), 'observed' if fresh_ocr else 'cached_context')
                elif self.stable('phase', new_phase, now, fresh_ocr, self.cfg['ocr_stable_samples']):
                    if old_phase is not None:
                        self.serial += 1
                        self.reset_support.clear()
                    accept(('inning', 'half'), 'resynchronized_after_gap' if old_phase is not None and new_phase > old_phase + 1 else 'inning_change' if old_phase is not None else 'observed')
                    self.resync_phase = None
                else:
                    hold(('inning', 'half'), 'waiting_for_confirmation', old_phase is not None)
            else:
                for key in ('inning', 'half'):
                    if vals[key] is None:
                        if raw[key] is not None:
                            hold((key,), 'invalid_value', True)
                    elif self.values[key] == vals[key]:
                        accept((key,), 'observed' if fresh_ocr else 'cached_context')
                    elif old_phase is not None:
                        hold((key,), 'partial_inning_change', True)
                    elif self.stable(key, vals[key], now, fresh_ocr, self.cfg['ocr_stable_samples']):
                        accept((key,))
                    else:
                        hold((key,), 'waiting_for_confirmation')

            for key in ('upper_team', 'lower_team', 'upper_score', 'lower_score'):
                value, old = vals[key], self.values[key]
                if value is None:
                    if raw[key] is not None:
                        hold((key,), 'invalid_value', key.endswith('_score'))
                    continue
                if key.endswith('_team') and old is not None and team_key(value) != team_key(old):
                    hold((key,), 'team_name_changed')
                elif key.endswith('_score') and old is not None and value < old:
                    hold((key,), 'score_decreased', True)
                elif old == value or (key.endswith('_team') and old is not None and team_key(value) == team_key(old)):
                    # Keep the first confirmed spelling for normalized aliases.
                    vals[key] = old
                    accept((key,), 'observed' if fresh_ocr else 'cached_context')
                elif self.stable(key, value, now, fresh_ocr, self.cfg['ocr_stable_samples']):
                    accept((key,))
                    if key.endswith('_score') and old is not None and value > old:
                        self.reset_support['score_increased'] = now
                else:
                    hold((key,), 'waiting_for_confirmation', old is not None and key.endswith('_score'))

            old, value = self.values['outs'], vals['outs']
            if value is not None:
                boundary = self.serial != self.outs_serial
                if old is not None and value < old and not boundary:
                    hold(('outs',), 'outs_decreased_without_inning_change', True)
                elif value == old or self.stable('outs', value, now, True, self.cfg['min_stable_samples']):
                    accept(('outs',), 'inning_change' if boundary else 'observed')
                    if value != old or not boundary:
                        self.outs_serial = self.serial
                    if old is not None and value > old and not boundary:
                        self.reset_support['outs_increased'] = now
                else:
                    hold(('outs',), 'waiting_for_confirmation', True)
            elif raw['outs'] is not None:
                hold(('outs',), 'invalid_value', True)

            runners = tuple(vals[k] for k in RUNNER_FIELDS)
            old_runners = tuple(self.values[k] for k in RUNNER_FIELDS)
            if all(v is not None for v in runners):
                boundary = self.serial != self.runner_serial
                drop = all(v is not None for v in old_runners) and sum(runners) < sum(old_runners)
                if drop and not boundary and not self.reset_support:
                    hold(RUNNER_FIELDS, 'runner_drop_without_out_or_run_evidence', True)
                elif runners == old_runners or self.stable('runners', runners, now, True, self.cfg['min_stable_samples']):
                    accept(RUNNER_FIELDS, 'inning_change' if boundary else 'observed')
                    if runners != old_runners or not boundary:
                        self.runner_serial = self.serial
                    if all(v is not None for v in old_runners) and sum(runners) > sum(old_runners):
                        self.reset_support['runner_added'] = now
                else:
                    hold(RUNNER_FIELDS, 'waiting_for_confirmation')

            pair = tuple(vals[k] for k in ('balls', 'strikes'))
            old_pair = tuple(self.values[k] for k in ('balls', 'strikes'))
            if all(v is not None for v in pair):
                boundary = self.serial != self.count_serial
                decrease = all(v is not None for v in old_pair) and any(a < b for a, b in zip(pair, old_pair))
                reset = pair in ((0, 0), (1, 0), (0, 1)) and bool(self.reset_support)
                if decrease and not boundary and not reset:
                    hold(('balls', 'strikes'), 'count_decreased_without_batter_change_evidence', True)
                elif blocked:
                    hold(('balls', 'strikes'), 'waiting_for_confirmation', True)
                elif pair == old_pair or self.stable('bs_pair', pair, now, True, self.cfg['min_stable_samples']):
                    accept(('balls', 'strikes'), 'inning_change' if boundary else 'possible_batter_change' if decrease else 'observed')
                    if pair != old_pair or not boundary:
                        self.count_serial = self.serial
                    if pair != old_pair:
                        self.reset_support.clear()
                else:
                    hold(('balls', 'strikes'), 'waiting_for_confirmation', True)
            elif any(raw[k] is not None and vals[k] is None for k in ('balls', 'strikes')):
                hold(('balls', 'strikes'), 'invalid_value', True)

        # A missing/held count may be displayed in the record, but is excluded
        # from count comparisons used to decide whether VideoMAE should run.
        comparable = on_screen and not blocked and all(info[k]['status'] == 'observed' for k in COUNT_FIELDS)
        for key in FIELDS:
            info[key]['source_age_sec'] = now - self.observed_at[key] if key in self.observed_at else None
            if info[key]['status'] in ('held', 'unknown'):
                self.audit.append(dict(frame=row.get('frame'), time_sec=now, field=key,
                                       raw_value=raw[key], recorded_value=self.values[key],
                                       **info[key], count_comparison_allowed=comparable))
        result['count'] = {k: self.values[k] for k in COUNT_FIELDS} if any(self.values[k] is not None for k in COUNT_FIELDS) else None
        result['runners'] = {k[7:]: self.values[k] for k in RUNNER_FIELDS} if any(self.values[k] is not None for k in RUNNER_FIELDS) else None
        result['scoreboard'] = dict(result.get('scoreboard') or {}, **{k: self.values[k] for k in BOARD_FIELDS})
        result['rule_validation'] = dict(count_comparison_allowed=comparable, fields=info,
                                         review_required=any(v['reason'] not in ('observed', 'cached_context', 'missing_or_unreadable', 'no_supported_scoreboard') for v in info.values()))
        return result


def filter_observations(rows, settings=None):
    validator = RuleFilter(settings)
    return [validator.apply(row) for row in rows], validator.audit
