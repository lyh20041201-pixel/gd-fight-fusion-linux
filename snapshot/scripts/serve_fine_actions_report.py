"""Local-only, allowlisted report/video viewer. Never exposes project files."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from scripts.prepare_fine_labels_bc import OUT,read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8011);args=parser.parse_args()
    # Only reviewed test videos and these specific generated report files.
    allowed={'/':OUT/'index.html','/index.html':OUT/'index.html','/REPORT.md':OUT/'REPORT.md'}
    allowed.update({'/video/'+r['id']:Path(r['path']) for r in read(OUT/'fine_annotations.json')['videos'] if r['split']=='test'})

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):self.send_file(False)
        def do_GET(self):self.send_file(True)
        def send_file(self,body):
            path=allowed.get(urlsplit(self.path).path)
            if path is None or not path.is_file():self.send_error(404);return
            size=path.stat().st_size;start=0;end=size-1;status=200
            range_header=self.headers.get('Range')
            if range_header:
                match=re.fullmatch(r'bytes=(\d*)-(\d*)',range_header)
                if not match or not any(match.groups()):self.send_error(416);return
                a,b=match.groups()
                if a:start=int(a);end=min(int(b),size-1) if b else size-1
                else:start=max(0,size-int(b))
                if start>end or start>=size:self.send_response(416);self.send_header('Content-Range',f'bytes */{size}');self.end_headers();return
                status=206
            self.send_response(status)
            content_type=mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
            if path.suffix in ['.html','.md']:content_type=('text/html' if path.suffix=='.html' else 'text/plain')+'; charset=utf-8'
            self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(end-start+1))
            self.send_header('Accept-Ranges','bytes');self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            if body:
                try:
                    with path.open('rb') as f:
                        f.seek(start);remaining=end-start+1
                        while remaining:
                            chunk=f.read(min(65536,remaining))
                            if not chunk:break
                            self.wfile.write(chunk);remaining-=len(chunk)
                except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
        def log_message(self,format,*args):pass

    print(f'http://127.0.0.1:{args.port}/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()
