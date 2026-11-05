import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import save_json,sha256,metadata


def ffmpeg_executable(explicit=None):
    if explicit:
        path=Path(explicit).expanduser().resolve()
        if not path.is_file():raise FileNotFoundError(path)
        return str(path)
    found=shutil.which('ffmpeg')
    if found:return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(f'FFmpegが必要です。PowerShellでは & "{sys.executable}" -m pip install imageio-ffmpeg を実行してください') from exc


def export_clip(video,output,begin,end,ffmpeg,source_sha256=None):
    output=Path(output);marker=output.with_suffix('.json')
    expected=dict(source_sha256=source_sha256 or sha256(video),start_sec=begin,end_sec=end,codec='libx264+aac')
    if output.is_file() and marker.is_file() and json.loads(marker.read_text())==expected:
        return
    if output.exists():raise FileExistsError(f'異なる条件の出力があります: {output}')
    temporary=output.with_name(output.stem+'.partial.mp4')
    command=[ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-y','-ss',f'{begin:.9f}','-i',str(video),
             '-t',f'{end-begin:.9f}','-map','0:v:0','-map','0:a?','-c:v','libx264','-preset','fast','-crf','20',
             '-c:a','aac','-movflags','+faststart',str(temporary)]
    try:
        subprocess.run(command,check=True)
        info=metadata(temporary)
        if abs(info['duration_sec']-(end-begin))>max(.15,2/info['fps']):raise RuntimeError('書き出した動画の長さが一致しません')
        temporary.replace(output);save_json(marker,expected)
    finally:
        if temporary.exists():temporary.unlink()


def run(job):
    from reference_matcher import index_video,locate
    root=Path(job['output']);ffmpeg=ffmpeg_executable(job.get('ffmpeg'));started=time.monotonic()
    source=index_video(job['video'],root/'image_index/source')
    results=[];clips=root/'scenes';clips.mkdir(exist_ok=True)
    for ref in job['references']:
        path=Path(job['references_dir'])/ref['filename']
        digest=sha256(path)
        if digest!=ref['sha256']:
            raise ValueError(f'{ref["filename"]} が登録した参考動画と異なります。投球時刻・ラベルの誤流用を防ぐため停止しました')
        print('照合:',ref['filename'],flush=True)
        index=index_video(path,root/'image_index/references'/path.stem)
        match=locate(job['video'],path,source,index,job['minimum_similarity'],job['ambiguity_margin'])
        match.update(reference=ref['filename'],reference_sha256=digest,reference_duration_sec=ref['duration_sec'])
        if match['status']=='matched':
            name=path.stem+'_from_source.mp4'
            export_clip(job['video'],clips/name,max(0,match['start_sec']-.3),min(source['identity']['duration_sec'],match['end_sec']+.3),ffmpeg,source['identity']['source_sha256'])
            match['scene_file']='scenes/'+name
        results.append(match);save_json(root/'matches.json',results)
        print(f'  {match["status"]} '+(f'{match["start_sec"]:.3f}〜{match["end_sec"]:.3f}秒' if 'start_sec' in match else ''),flush=True)
    with (root/'matches.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['reference','status','start_sec','end_sec','similarity','alternative_similarity','scene_file'],extrasaction='ignore')
        writer.writeheader();writer.writerows(results)
    save_json(root/'reference_summary.json',dict(source=source['identity'],requested=len(results),
              matched=sum(r['status']=='matched' for r in results),ambiguous=sum(r['status']=='ambiguous' for r in results),
              not_found=sum(r['status']=='not_found' for r in results),elapsed_sec=time.monotonic()-started,
              method='Ordered visual reference matching; these scores measure matching, not foul probability.'))
    width=1100;height=55+len(results)*32;duration=source['identity']['duration_sec']
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="照合した場面の元動画内の位置">',
         '<text x="180" y="20" font-size="14">元動画の中で照合できた場面の位置</text>']
    for i,r in enumerate(results):
        y=45+i*32
        svg.append(f'<text x="0" y="{y+12}" font-size="13">{html.escape(r["reference"])}</text>')
        if r['status']=='matched':
            x=180+880*r['start_sec']/duration;w=max(2,880*(r['end_sec']-r['start_sec'])/duration)
            svg.append(f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="19" fill="#3888b5"/>')
        else:svg.append(f'<text x="180" y="{y+12}" font-size="13">{r["status"]}</text>')
    svg.append('</svg>');(root/'match_timeline.svg').write_text(''.join(svg),encoding='utf-8')
    page='<!doctype html><html lang="ja"><meta charset="utf-8"><title>参考場面の抽出結果</title><style>body{font:17px/1.7 sans-serif;max-width:1000px;margin:30px auto}video{width:100%;max-height:480px}article{border:1px solid #ccc;padding:18px;margin:18px 0}</style><h1>10本の参考場面との映像照合</h1><p>一致度は映像の照合値です。ファウル確率とは異なります。見つからない場面や重複する場面も記録します。</p><a href="matches.csv">時刻CSV</a>'
    page+='<img style="width:100%" src="match_timeline.svg" alt="元動画内で照合した場面の位置">'
    if job['analyze']:page+='<p><a href="analysis/index.html">BSO・VideoMAEによる確認結果</a></p>'
    for r in results:
        page+='<article><h2>'+html.escape(r['reference'])+'</h2><p>'+r['status']+'</p>'
        if r.get('scene_file'):
            page+=f'<p>{r["start_sec"]:.3f}〜{r["end_sec"]:.3f}秒</p><video controls preload="metadata" src="{r["scene_file"]}"></video><p><a download href="{r["scene_file"]}">MP4を保存</a></p>'
        page+='</article>'
    (root/'index.html').write_text(page+'</html>',encoding='utf-8')
    if job['analyze']:
        from analyze_matched_scenes import analyze
        analyze(job,results,source['identity'],ffmpeg)
    print('結果:',root/'index.html',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job');args=parser.parse_args()
    run(json.loads(Path(args.job).read_text(encoding='utf-8')))
