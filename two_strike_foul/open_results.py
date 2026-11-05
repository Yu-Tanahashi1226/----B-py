import argparse
from pathlib import Path
import webbrowser

parser=argparse.ArgumentParser(description='ローカルに生成された結果をブラウザーで開きます')
parser.add_argument('result_directory')
args=parser.parse_args()
page=Path(args.result_directory).resolve()/'index.html'
if not page.is_file():raise SystemExit(f'結果ページがありません: {page}')
webbrowser.open(page.as_uri())
print(page)
