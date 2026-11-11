"""Find the ten known scenes in a full video, export MP4, optionally verify counts/motion/outcome."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def load_config(path, required=False):
    path=Path(path).resolve()
    if not path.is_file():
        if required:
            raise FileNotFoundError(f'設定ファイルが見つかりません: {path}\n'
                                    '先に configure_windows.py を正常終了させ、config.local.json が保存されたことを確認してください。'
                                    '重みが見つからず設定が停止した場合は、その重みを配置して設定をやり直してください。')
        return None
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',required=True)
    parser.add_argument('--references-dir',required=True,help='10本の参考動画を置いたフォルダー')
    parser.add_argument('--output',required=True)
    parser.add_argument('--config',help='既存のconfig.local.json。YOLO環境のPythonを再利用')
    parser.add_argument('--python',help='OpenCVとNumPyを使えるPython。--configより優先')
    parser.add_argument('--only',nargs='+',help='指定した参考動画だけを探す（例：5urafaul8.mp4）')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--analyze',action='store_true',help='抽出に加えてBSO・投球動作・試作ファウル判別を実行')
    parser.add_argument('--scoreboard-scope',choices=['full','matched'],default='full',help='全体を記録、または照合した場面と前後だけを記録')
    parser.add_argument('--ffmpeg',help='ffmpeg実行ファイルの場所')
    parser.add_argument('--minimum-similarity',type=float,default=.90)
    parser.add_argument('--ambiguity-margin',type=float,default=.025)
    args=parser.parse_args()
    if not 0<args.minimum_similarity<=1 or not 0<=args.ambiguity_margin<1:parser.error('Invalid similarity thresholds')
    config=Path(args.config).resolve() if args.config else ROOT/'config.local.json'
    cfg=load_config(config,required=bool(args.config or args.analyze))
    video=Path(args.video).expanduser().resolve();refs=Path(args.references_dir).expanduser().resolve()
    if not video.is_file():raise FileNotFoundError(video)
    catalog=json.loads((ROOT/'reference_catalog.json').read_text(encoding='utf-8'))
    selected=[r for r in catalog['references'] if args.only is None or r['filename'] in args.only]
    if not selected or (args.only and set(args.only)-{r['filename'] for r in selected}):parser.error('--only に指定された参考動画が登録されていません')
    missing=[r['filename'] for r in selected if not (refs/r['filename']).is_file()]
    if missing:raise FileNotFoundError('参考動画がありません: '+', '.join(missing))
    executable=args.python or (cfg['python']['yolo'] if cfg else sys.executable)
    if '/' in executable or '\\' in executable:
        p=Path(executable).expanduser();executable=os.path.abspath(p if p.is_absolute() else config.parent/p)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=args.resume)
    identity=dict(video=str(video),references_dir=str(refs),output=str(output),references=selected,
                  catalog_source=catalog['annotation_source'],config=str(config) if cfg else None,
                  analyze=args.analyze,scoreboard_scope=args.scoreboard_scope,ffmpeg=args.ffmpeg,
                  minimum_similarity=args.minimum_similarity,ambiguity_margin=args.ambiguity_margin)
    path=output/'reference_job.json'
    if args.resume and path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=identity:raise ValueError('再開時の設定が前回と異なります。新しい出力先を使ってください')
    path.write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
    env=dict(os.environ,PYTHONIOENCODING='utf-8');env.setdefault('OPENBLAS_NUM_THREADS','2');env.setdefault('OMP_NUM_THREADS','2')
    code=subprocess.run([executable,str(ROOT/'reference_worker.py'),str(path)],env=env).returncode
    return code


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        print(f'停止: {exc}',file=sys.stderr);raise SystemExit(1)
