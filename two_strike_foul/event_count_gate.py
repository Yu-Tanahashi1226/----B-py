"""Compare count before/after a reference-localized pitch, without a disappearance requirement."""
from gating import stable_runs,summarize,COUNT_KEYS


def count_gate(rows,start_sec,end_sec,next_start_sec=None,period=.5):
    pre=stable_runs(rows,max(0,start_sec-5),start_sec-.05,period,samples=2)
    after_end=min(end_sec+8,next_start_sec-.5) if next_start_sec is not None else end_sec+8
    post=stable_runs(rows,end_sec+.5,after_end,period,samples=2)
    before=summarize(pre[-1]) if pre else None
    after=summarize(post[-1]) if post else None
    states=[summarize(run) for run in post]
    changed=[state for state in states if before and tuple(state[k] for k in COUNT_KEYS)!=tuple(before[k] for k in COUNT_KEYS)]
    ignored=[]
    # Outs cannot fall within the same half inning. Only ignore temporary drops
    # when the ORIGINAL count returns with >=3 consistent samples, B/S never
    # changed, and inning+half are explicitly known and equal. Raw rows remain
    # untouched and the rejected readings are reported for annotation review.
    if before and after and after['support']>=3 and tuple(before[k] for k in COUNT_KEYS)==tuple(after[k] for k in COUNT_KEYS):
        for state in changed:
            same_half=all(before.get(k) is not None and state.get(k)==before[k] and after.get(k)==before[k] for k in ['inning','half'])
            if same_half and state['balls']==before['balls'] and state['strikes']==before['strikes'] and state['outs']<before['outs']:
                ignored.append(state)
    remaining=[state for state in changed if state not in ignored]
    if before is None or start_sec-before['last_sec']>2.:
        status='before_bso_unknown'
    elif before['strikes']!=2:status='not_two_strikes'
    elif after is None:status='after_bso_unknown'
    elif any(before.get(k) is not None and after.get(k) is not None and before[k]!=after[k] for k in ['inning','half']):
        status='inning_or_half_changed_review'
    elif remaining:
        status='bso_changed_or_delayed_update'
    else:status='two_strikes_bso_unchanged'
    return dict(selected=status=='two_strikes_bso_unchanged',reason=status,before=before,after=after,
                post_count_states=states,ignored_inconsistent_out_drops=ignored,
                disappearance_required=False,confirmed_foul=None)
