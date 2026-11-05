"""Reproduce the small outcome pilot; target reference is always held out."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def train(vectors,output,evaluation,review=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    data=np.load(vectors,allow_pickle=False)
    X=data['vectors'];labels=data['approximate_labels'].copy();groups=data['references'];pitches=data['pitch_indices']
    if review:
        with Path(review).open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        corrected={}
        for row in rows:
            value=row['true_foul'].strip()
            if value not in ['0','1']:raise ValueError('確認済みtrue_foulを全投球に0または1で入力してください')
            key=(row['reference'],int(row['pitch_index']))
            if key in corrected:raise ValueError('重複する投球ラベルがあります')
            corrected[key]=int(value)
        for i,(group,pitch) in enumerate(zip(groups,pitches)):
            labels[i]=corrected[(str(group),int(pitch))]
    heads={};predictions=np.zeros(len(labels),dtype=int);scores=np.zeros(len(labels))
    for reference in sorted(set(groups)):
        train_mask=groups!=reference;test_mask=~train_mask
        if len(set(labels[train_mask]))<2:raise ValueError('各学習分割にファウル・非ファウルの両方が必要です')
        model=LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,solver='liblinear',random_state=0).fit(X[train_mask],labels[train_mask])
        predictions[test_mask]=model.predict(X[test_mask]);scores[test_mask]=model.predict_proba(X[test_mask])[:,1]
        heads[str(reference)]=dict(training_references=sorted(set(groups[train_mask].tolist())),coefficient=model.coef_[0].tolist(),intercept=float(model.intercept_[0]))
    payload=dict(schema_version=1,encoder='MCG-NJU/videomae-base-finetuned-kinetics',revision='488eb9a0565f257b32866000305c8178965eb9f6',
                 pooling='3 equal temporal bins; concatenate mean 768-D embeddings; L2 normalize',context_before_sec=.5,context_after_sec=4.,
                 training_labels='User-reviewed CSV' if review else 'Previous assistant visual review; approximate, not independently verified user ground truth.',
                 leave_one_reference_out_heads=heads)
    Path(output).write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    report=dict(samples=len(labels),correct=int((labels==predictions).sum()),confusion_matrix_true_rows_predicted_columns=confusion_matrix(labels,predictions,labels=[0,1]).tolist(),
                predictions=[dict(reference=str(g),pitch_index=int(p),label=int(y),decision=int(z),score=float(s)) for g,p,y,z,s in zip(groups,pitches,labels,predictions,scores)],
                limitation='Only one game, clip-held-out evaluation; not evidence of generalization to another game.')
    Path(evaluation).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser();parser.add_argument('--vectors',default=str(root/'reference_training_vectors.npz'))
    parser.add_argument('--output',default='reference_outcome_heads.reviewed.json');parser.add_argument('--evaluation',default='outcome_evaluation.reviewed.json')
    parser.add_argument('--labels-review');args=parser.parse_args()
    train(args.vectors,args.output,args.evaluation,args.labels_review)
