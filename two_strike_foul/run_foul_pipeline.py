"""Whole-video scoreboard -> unchanged-two-strike gate -> selected-window VideoMAE."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import save_json, sha256
from gating import make_gate
from scoreboard_rule_report import run_validation

ROOT = Path(__file__).resolve().parent


def resolve_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    for section, keys in [('yolo', ['weights', 'inner_weights']), ('ocr', ['detector_dir', 'recognizer_dir']), ('pitch', ['checkpoint'])]:
        for key in keys:
            target = Path(cfg[section][key]).expanduser()
            cfg[section][key] = str(target.resolve() if target.is_absolute() else (path.parent / target).resolve())
    for stage, executable in cfg['python'].items():
        # Plain "python" uses PATH; relative filesystem paths use config location.
        if '/' in executable or '\\' in executable:
            target = Path(executable).expanduser()
            # Resolving a venv's python symlink would launch the base interpreter
            # and silently lose that environment's installed packages.
            cfg['python'][stage] = os.path.abspath(target if target.is_absolute() else path.parent / target)
    return cfg


def preflight(cfg):
    errors = []
    for section, keys in [('yolo', ['weights', 'inner_weights']), ('ocr', ['detector_dir', 'recognizer_dir']), ('pitch', ['checkpoint'])]:
        for key in keys:
            if not Path(cfg[section][key]).exists():
                errors.append(f'{section}.{key}: {cfg[section][key]} がありません')
    dependencies = {'yolo': ['cv2', 'ultralytics', 'torch', 'matplotlib', 'numpy'], 'ocr': ['cv2', 'paddleocr', 'paddle'], 'pitch': ['cv2', 'torch', 'transformers', 'numpy']}
    for stage, modules in dependencies.items():
        script = 'import importlib.util,json; print(json.dumps({m:importlib.util.find_spec(m) is not None for m in ' + repr(modules) + '}))'
        try:
            check = subprocess.run([cfg['python'][stage], '-c', script], capture_output=True, text=True, timeout=30, check=True)
            installed = json.loads(check.stdout.strip())
            errors.extend(f'{stage}環境に {name} がありません' for name, available in installed.items() if not available)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(f'{stage}環境を起動できません: {exc}')
    if not 0 < cfg['sample_period_sec'] <= 2:
        errors.append('sample_period_sec は0より大きく2以下にしてください')
    return dict(ready=not errors, errors=errors, scope='scoreboard_gated_only')


def environment(output):
    env = dict(os.environ)
    for variable, name in [('XDG_CACHE_HOME', 'xdg'), ('PADDLE_PDX_CACHE_HOME', 'paddlex'), ('YOLO_CONFIG_DIR', 'ultralytics'), ('MPLCONFIGDIR', 'matplotlib')]:
        directory = output / '.cache' / name
        directory.mkdir(parents=True, exist_ok=True)
        env.setdefault(variable, str(directory))
    env.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    env.setdefault('OMP_NUM_THREADS', '2')
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    return env


def stage(job, name, script, arguments, env, resume):
    output = Path(job['output'])
    marker = output / f'{name}.done.json'
    source_files = ['common.py', script]
    source_files += ['pitch_backend.py'] if name == 'pitch' else ['board_adapter.py', 'ocr_fallback.py']
    fingerprint = dict(job_sha256=sha256(output / 'job.json'),
                       code={name:sha256(ROOT / name) for name in source_files})
    if name == 'pitch':
        fingerprint['gate_sha256'] = sha256(output / 'gate.json')
    if name == 'ocr':
        fingerprint['yolo_output_sha256'] = sha256(output / 'yolo_boards.jsonl')
    if resume and marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get('fingerprint') == fingerprint:
            print(f'再利用: {name}', flush=True)
            return
    print(f'開始: {name}（進行状況: {output / (name + ".log")}）', flush=True)
    with (output / f'{name}.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([job['config']['python'][name], str(ROOT / script), *arguments], stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f'実行中: {name}', flush=True)
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f'{name} が終了コード {code} で停止しました。{output / (name + ".log")} を確認してください')
    save_json(marker, dict(fingerprint=fingerprint, completed=True))
    print(f'完了: {name}', flush=True)


def execute(job, resume=False, stop_after=None):
    output = Path(job['output'])
    env = environment(output)
    job_path = str(output / 'job.json')
    stage(job, 'yolo', 'scoreboard_worker.py', ['yolo', job_path], env, resume)
    stage(job, 'ocr', 'scoreboard_worker.py', ['ocr', job_path], env, resume)
    rows = run_validation(job)
    print(f'野球ルール確認: {output / "rules" / "review.html"}', flush=True)
    gate = make_gate(rows, job['fps'], job['frames'], job['config']['sample_period_sec'], job['config']['gate'])
    save_json(output / 'gate.json', gate)
    print(f'条件に一致: {gate["selected_episodes"]}場面 / VideoMAE対象: {len(gate["windows"])}区間', flush=True)
    if stop_after == 'gate':
        return
    stage(job, 'pitch', 'pitch_worker.py', [job_path], env, resume)
    # Reports use cv2 for evidence images, so run them in the YOLO environment.
    subprocess.run([job['config']['python']['yolo'], str(ROOT / 'report.py'), job_path], env=env, check=True)
    save_json(output / 'completed.json', dict(status='completed', source_sha256=job['sha256'], scope='scoreboard_gated_only'))
    print(f'結果: {output / "index.html"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.windows.json'))
    parser.add_argument('--video', help='解析する動画。指定がなければ設定ファイルのvideoを使用')
    parser.add_argument('--output', help='新規結果フォルダー（省略時はoutput内に日時付きで作成）')
    parser.add_argument('--preflight', action='store_true', help='環境とファイルを確認して終了')
    parser.add_argument('--resume', help='中断した結果フォルダー。保存した設定で再開')
    parser.add_argument('--stop-after', choices=['gate'], help='VideoMAEを実行せず対象区間の決定まで行う')
    parser.add_argument('--include-following-pitch', action=argparse.BooleanOptionalAction, default=None,
                        help='条件一致の直後の投球も比較用に調べる（既定では調べない）')
    parser.add_argument('--references-dir',help='10本の参考動画を照合し、MP4抽出とBSO・VideoMAE確認を行う新モード')
    parser.add_argument('--scoreboard-scope',choices=['full','matched'],default='full',help='参考場面モードのYOLO/OCR対象範囲')
    args = parser.parse_args()
    if args.references_dir:
        if not args.video or not args.output:
            parser.error('参考場面モードは --video と --output を指定してください')
        if args.resume or args.stop_after or args.preflight or args.include_following_pitch is not None:
            parser.error('参考場面モードの再開・抽出だけの実行は extract_reference_scenes.py の手順を参照してください')
        return subprocess.run([sys.executable,str(ROOT/'extract_reference_scenes.py'),'--config',args.config,
                               '--video',args.video,'--references-dir',args.references_dir,'--output',args.output,
                               '--analyze','--scoreboard-scope',args.scoreboard_scope]).returncode
    if args.resume:
        if args.video or args.output or args.include_following_pitch is not None:
            parser.error('--resume は保存済みの動画・設定で再開します。変更する場合は新規実行してください')
        output = Path(args.resume).resolve()
        job = json.loads((output / 'job.json').read_text(encoding='utf-8-sig'))
        if Path(job['output']).resolve() != output:
            raise ValueError('結果フォルダーが移動しています。保存したoutputパスを確認してください')
        if sha256(job['video']) != job['sha256']:
            raise ValueError('元動画が前回と異なります。新規実行してください')
        cfg = job['config']
    else:
        cfg = resolve_config(args.config)
        if args.include_following_pitch is not None:
            cfg['gate']['include_following_pitch'] = args.include_following_pitch
    check = preflight(cfg)
    if args.preflight or not check['ready']:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return 0 if check['ready'] else 2
    if not args.resume:
        video = Path(args.video or cfg.get('video', 'test.mp4')).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f'動画がありません: {video}')
        probe = subprocess.run([cfg['python']['yolo'], str(ROOT / 'scoreboard_worker.py'), 'metadata', str(video)],
                               capture_output=True, text=True, check=True)
        info = json.loads(probe.stdout.strip())
        output = Path(args.output).resolve() if args.output else ROOT / 'output' / f'{video.stem}_{time.strftime("%Y%m%d_%H%M%S")}'
        output.mkdir(parents=True, exist_ok=False)
        period = cfg['sample_period_sec']
        frames = sorted(set(min(info['frames']-1, round(i*period*info['fps'])) for i in range(int(info['duration_sec']/period)+1)))
        job = dict(video=str(video), output=str(output), sha256=sha256(video), sample_frames=frames, config=cfg, **info)
        # Model hashes are part of the resumable job identity.
        job['model_hashes'] = {name: sha256(cfg['yolo'][name]) for name in ['weights', 'inner_weights']}
        job['model_hashes']['checkpoint'] = sha256(cfg['pitch']['checkpoint'])
        save_json(output / 'job.json', job)
    else:
        current = {name: sha256(cfg['yolo'][name]) for name in ['weights', 'inner_weights']}
        current['checkpoint'] = sha256(cfg['pitch']['checkpoint'])
        if current != job['model_hashes']:
            raise ValueError('前回から重みが変わっています。新規実行してください')
    execute(job, bool(args.resume), args.stop_after)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'停止: {exc}', file=sys.stderr)
        raise SystemExit(1)
