"""Find the user's existing model assets and Conda environments; no installations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def choose_file(candidates, label):
    for path in candidates:
        if path.is_file():return str(path.resolve())
    raise FileNotFoundError(label+' が見つかりません。確認した場所:\n'+'\n'.join(str(p) for p in candidates))


def python_candidates():
    candidates=[Path(sys.executable)]
    conda=os.environ.get('CONDA_EXE') or shutil.which('conda')
    if conda:
        try:
            info=json.loads(subprocess.run([conda,'info','--json'],capture_output=True,text=True,check=True,timeout=30).stdout)
            roots=[Path(info['root_prefix']),*[Path(p) for p in info.get('envs',[])]]
            for directory in info.get('envs_dirs',[]):
                location=Path(directory)
                if location.exists():roots.extend(p for p in location.iterdir() if p.is_dir())
            candidates.extend(root / ('python.exe' if os.name=='nt' else 'bin/python') for root in roots)
        except (OSError,ValueError,subprocess.SubprocessError):pass
    return list(dict.fromkeys(str(p.absolute()) for p in candidates if p.is_file()))


def find_python(candidates, modules, preferred, explicit=None):
    if explicit:
        candidates=[str(Path(explicit).expanduser().absolute())]
    else:
        candidates=sorted(candidates,key=lambda p:next((i for i,name in enumerate(preferred) if name in Path(p).parts),len(preferred)))
    for executable in candidates:
        script='import importlib.util; raise SystemExit(0 if all(importlib.util.find_spec(m) for m in '+repr(modules)+') else 1)'
        try:
            if subprocess.run([executable,'-c',script],capture_output=True,timeout=30).returncode==0:
                return executable
        except (OSError,subprocess.TimeoutExpired):pass
    raise RuntimeError('対応するPythonが見つかりません: '+','.join(modules)+'。READMEの環境作成手順を参照してください')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True,help='for_labelimgフォルダー')
    parser.add_argument('--pitch-checkpoint',help='既存compact_actionformer.pthのパス')
    parser.add_argument('--scoreboard-weights')
    parser.add_argument('--inner-weights')
    parser.add_argument('--yolo-python')
    parser.add_argument('--ocr-python')
    parser.add_argument('--pitch-python')
    parser.add_argument('--ocr-model-root',help='PP-OCRv6_medium_det/recが入ったフォルダー')
    parser.add_argument('--output',default=str(ROOT/'config.local.json'))
    args=parser.parse_args()
    project=Path(args.project).expanduser().resolve()
    cfg=json.loads((ROOT/'config.windows.json').read_text(encoding='utf-8'))
    board=[Path(args.scoreboard_weights)] if args.scoreboard_weights else [project/'runs/detect/scoreboard_model-2/weights/best.pt',project/'models/scoreboard_best.pt']
    inner=[Path(args.inner_weights)] if args.inner_weights else [project/'runs/detect/bso_inner_no_empty-3/weights/best.pt',project/'runs/detect/bso_inner_model/weights/best.pt',project/'models/inner_best.pt']
    pitch=[Path(args.pitch_checkpoint)] if args.pitch_checkpoint else [project/'compact_actionformer.pth',project/'models/compact_actionformer.pth',project.parent/'VideoMAE-Baseball/results/videomae_actionformer/compact_actionformer.pth']
    cfg['yolo']['weights']=choose_file(board,'スコアボード重み')
    cfg['yolo']['inner_weights']=choose_file(inner,'BSO・走者重み')
    cfg['pitch']['checkpoint']=choose_file(pitch,'Compact ActionFormer重み')
    modelroot=Path(args.ocr_model_root).expanduser().resolve() if args.ocr_model_root else Path.home()/'.paddlex/official_models'
    for key,name in [('detector','PP-OCRv6_medium_det'),('recognizer','PP-OCRv6_medium_rec')]:
        directory=modelroot/name
        if not (directory/'inference.pdiparams').is_file():raise FileNotFoundError(f'PaddleOCRモデルがありません: {directory}')
        cfg['ocr'][key+'_dir']=str(directory)
    candidates=python_candidates()
    cfg['python']['yolo']=find_python(candidates,['ultralytics','cv2','torch','matplotlib','numpy'],['yolo','base'],args.yolo_python)
    cfg['python']['ocr']=find_python(candidates,['paddle','paddleocr','cv2'],['paddleocr','baseball-ocr','ocr','base'],args.ocr_python)
    cfg['python']['pitch']=find_python(candidates,['torch','transformers','cv2','numpy'],['actionformer','videomae'],args.pitch_python)
    cfg['video']=str(project/'test.mp4')
    output=Path(args.output).resolve()
    output.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    print('設定を保存:',output)
    print('使うPython:',json.dumps(cfg['python'],ensure_ascii=False,indent=2))
    print('次: python run_foul_pipeline.py --config "'+str(output)+'" --preflight')


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError) as exc:
        print(str(exc),file=sys.stderr);raise SystemExit(1)
