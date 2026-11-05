"""Compare manually completed review.csv fields with saved predictions."""
import argparse
import csv
import json
from pathlib import Path
from collections import Counter


def boolean(value):
    table={'true':True,'1':True,'yes':True,'false':False,'0':False,'no':False}
    return table[value.strip().lower()]


def audit(path):
    totals=Counter();errors=Counter();examples=[]
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            count=json.loads(row['predicted_bso']) or {}
            runners=json.loads(row['predicted_runners']) or {}
            predictions={'board_present':boolean(row['predicted_board_present']),**count,**runners}
            for field in ['board_present','balls','strikes','outs','first','second','third']:
                raw=row.get('true_'+field,'').strip()
                if not raw:continue
                try:truth=int(raw) if field in ['balls','strikes','outs'] else boolean(raw)
                except (ValueError,KeyError) as exc:raise ValueError(f'frame {row["frame"]}: true_{field} の値が不正です: {raw}') from exc
                if field in ['balls','strikes','outs'] and not 0<=truth<=dict(balls=3,strikes=2,outs=2)[field]:
                    raise ValueError(f'frame {row["frame"]}: true_{field} が範囲外です')
                totals[field]+=1
                if predictions.get(field)!=truth:
                    errors[field]+=1
                    examples.append(dict(frame=row['frame'],field=field,predicted=predictions.get(field),truth=truth,image=row['image']))
    return dict(reviewed_fields=dict(totals),mismatches=dict(errors),examples=examples,
                selected_sample_only=True,interpretation='選ばれた確認画像内の不一致です。動画全体の精度や学習データ不足を示す値ではありません。')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('review_csv');args=parser.parse_args()
    result=audit(args.review_csv)
    output=Path(args.review_csv).with_name('manual_audit.json')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
