"""Local browser player using exactly the corrected PPO Tetris environment."""
import argparse
import json
import os
from pathlib import Path
import signal
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
import webbrowser

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from human_demonstrations import HumanRecorder, ROOT, new_session_path


def make_handler(recorder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, value, status=200, content_type='application/json'):
            data = value.encode() if isinstance(value, str) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Recording/atomic autosave remains valid after a tab closes.

        def do_GET(self):
            request = urlparse(self.path)
            if request.path == '/':
                self.respond((ROOT/'human_recorder.html').read_text(), content_type='text/html; charset=utf-8')
            elif request.path == '/api/status':
                owner = parse_qs(request.query).get('owner', [''])[0]
                self.respond(recorder.status(owner))
            else:
                self.respond({'error': 'Not found'}, 404)

        def do_POST(self):
            origin = self.headers.get('Origin')
            if origin and origin != 'http://'+self.headers.get('Host', ''):
                self.respond({'error': 'Only the local player can control this recorder.'}, 403)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('Invalid request size.')
                payload = json.loads(self.rfile.read(length))
                owner = payload.get('owner')
                if not isinstance(owner, str) or not 1 <= len(owner) <= 100:
                    raise ValueError('Missing player identity.')
                if self.path == '/api/start':
                    result = recorder.start(owner)
                elif self.path == '/api/step':
                    result = recorder.step(payload['action'], payload['step'], owner,
                                           payload['episode_id'], payload.get('speed', 1.))
                elif self.path == '/api/finish':
                    recorder.finish('manual_stop', owner)
                    result = recorder.status(owner)
                else:
                    self.respond({'error': 'Not found'}, 404); return
                self.respond(result)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond({'error': str(exc)}, 409)
            except Exception:
                import traceback
                traceback.print_exc()
                self.respond({'error': 'Recording paused. Check the terminal; use Save & end or Ctrl+C to save.'}, 500)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='Fresh session directory; defaults to runs/tetris/human_demos/<timestamp>.')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--seed-start', type=int, default=200000)
    parser.add_argument('--no-open', action='store_true', help='Print URL without opening a browser.')
    parser.add_argument('--source', choices=['human', 'automated_test'], default='human', help=argparse.SUPPRESS)
    args = parser.parse_args()
    recorder = HumanRecorder(args.out or new_session_path(), args.seed_start, source=args.source)
    print('Preparing the corrected Tetris emulator (first start compiles JAX)…', flush=True)
    recorder.warmup()
    server = HTTPServer(('127.0.0.1', args.port), make_handler(recorder))
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'PLAY: {url}\nSAVE DIRECTORY: {recorder.directory.resolve()}\nCtrl+C saves the current game and stops the server.', flush=True)
    if not args.no_open:
        webbrowser.open(url)
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        recorder.finish('server_shutdown')
        server.server_close()


if __name__ == '__main__':
    main()
