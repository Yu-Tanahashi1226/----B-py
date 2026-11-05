"""Count existing YOLO labels without treating missing labels as negatives."""
import argparse
import json
from pathlib import Path
from collections import Counter


def inventory(directory, classes):
    root=Path(directory).expanduser().resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    totals=Counter();positive_images=Counter();invalid=[];empty=0;files=0
    paths=list((root/'labels').rglob('*.txt')) if (root/'labels').is_dir() else []
    for path in paths:
        files+=1;seen=set();lines=path.read_text(encoding='utf-8-sig').splitlines()
        if not any(line.strip() for line in lines):empty+=1
        for index,line in enumerate(lines,1):
            if not line.strip():continue
            values=line.split()
            try:
                if len(values)!=5:raise ValueError('5列の検出用YOLO形式が必要です')
                identity=int(values[0]);x,y,w,h=map(float,values[1:])
                if not 0<=identity<len(classes):raise ValueError('class IDが範囲外です')
                if not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):raise ValueError('正規化座標が不正です')
                totals[classes[identity]]+=1;seen.add(classes[identity])
            except ValueError as exc:invalid.append(dict(file=str(path.relative_to(root)),line=index,reason=str(exc)))
        positive_images.update(seen)
    images=[p for p in (root/'images').rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp']]
    unlabeled=[]
    for image in images:
        relative=image.relative_to(root/'images').with_suffix('.txt')
        if not (root/'labels'/relative).exists():unlabeled.append(str(image.relative_to(root)))
    return dict(root=str(root),classes=classes,label_files=files,images=len(images),empty_label_files=empty,
                boxes_by_class={c:totals[c] for c in classes},positive_images_by_class={c:positive_images[c] for c in classes},
                images_without_label_files=unlabeled,invalid_labels=invalid,
                interpretation='未ラベル画像を負例とはみなしません。枚数だけでは不足を判断できません。失敗画像との対応と、試合単位の分離評価が必要です。')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--kind',choices=['scoreboard','inner'],required=True)
    parser.add_argument('--output',default='dataset_inventory.json');args=parser.parse_args()
    result=inventory(args.dataset,['scoreboard'] if args.kind=='scoreboard' else ['ball','strike','out','runner'])
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='images_without_label_files'},ensure_ascii=False,indent=2))
