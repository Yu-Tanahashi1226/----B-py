"""Run the same pipeline independently on a folder of short clips."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--input-dir',required=True)
    parser.add_argument('--pattern',default='*faul*.mp4')
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    videos=sorted(Path(args.input_dir).resolve().glob(args.pattern))
    if not videos:raise SystemExit('一致する動画がありません')
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=args.resume)
    results=[]
    for video in videos:
        destination=output/video.stem
        command=[sys.executable,str(ROOT/'run_foul_pipeline.py')]
        if args.resume and (destination/'job.json').is_file():
            command+=['--resume',str(destination)]
        else:
            command+=['--config',str(Path(args.config).resolve()),'--video',str(video),'--output',str(destination)]
        print('解析:',video.name,flush=True)
        try:
            code=subprocess.run(command,check=False).returncode
            result=dict(video=video.name,output=str(destination),exit_code=code)
        except OSError as exc:
            result=dict(video=video.name,output=str(destination),exit_code=1,error=str(exc))
        results.append(result)
        (output/'batch_status.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    return 1 if any(r['exit_code'] for r in results) else 0


if __name__=='__main__':raise SystemExit(main())
