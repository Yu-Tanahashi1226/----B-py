"""Small, explicit VideoMAE outcome head; no unsafe pickle/model loading."""
import numpy as np


def pool_context(features, centers_sec, begin, end):
    if end<=begin or len(features)!=len(centers_sec):raise ValueError('Invalid outcome context')
    pooled=[]
    edges=np.linspace(begin,end,4)
    for a,b in zip(edges,edges[1:]):
        indices=np.flatnonzero((centers_sec>=a)&(centers_sec<b))
        if not len(indices):indices=np.asarray([int(np.argmin(abs(centers_sec-(a+b)/2)))])
        pooled.append(features[indices].mean(axis=0))
    vector=np.concatenate(pooled).astype(np.float32)
    return vector/max(float(np.linalg.norm(vector)),1e-8)


def predict_outcome(payload,reference,vector):
    head=payload['leave_one_reference_out_heads'][reference]
    if reference in head['training_references']:raise ValueError('Reference leakage in the outcome head')
    weights=np.asarray(head['coefficient'],dtype=np.float64)
    if weights.shape!=vector.shape:raise ValueError('Outcome feature shape mismatch')
    logit=float(weights@vector+head['intercept'])
    score=float(1/(1+np.exp(-np.clip(logit,-40,40))))
    return dict(foul_model_score=score,foul_model_decision=score>=.5,
                training_excludes_this_reference=True,confirmed_foul=None,
                interpretation='Experimental classifier score; not a calibrated foul probability.')
