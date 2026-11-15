""" All-leagues dashboard: a small local web server (standard library only) showing the predictions of every league,
    the preferred model per league & market, the live track record of stored predictions and the saved models.

    Run with: python cli.py dashboard   (or the "Dashboard" button of the GUI)
    It only listens on 127.0.0.1.
"""

import json
import math
import os
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import date, datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIRECTORY = Path(__file__).resolve().parent / 'static'
CACHE_SECONDS = 30
JOB_COMMANDS = {
    'refresh': ['refresh', '--all'],    # download results & settle stored predictions
    'predict': ['predict', '--all'],    # predict upcoming fixtures of every league
}


def _clean(value: Any) -> Any:
    """ Converts values to JSON-safe types (NaN -> None). """

    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if hasattr(value, 'item'):
        return _clean(value.item())
    return value


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [{key: _clean(value) for key, value in row.items()} for row in df.to_dict(orient='records')]


class DashboardData:
    """ Loads (and briefly caches) the dashboard data from the project's storage. """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def _cached(self, key: str, loader):
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None and time.time() - entry[0] < CACHE_SECONDS:
                return entry[1]
        value = loader()
        with self._lock:
            self._cache[key] = (time.time(), value)
        return value

    def invalidate(self):
        with self._lock:
            self._cache.clear()

    def _scores(self) -> pd.DataFrame:
        from src.core.automation.insights import model_scores
        from src.core.automation.ledger import PredictionLedger

        return self._cached('scores', lambda: model_scores(ledger=PredictionLedger()))

    def leagues(self) -> List[Dict[str, Any]]:
        def load():
            from src.core.database.league import LeagueDatabase
            from src.core.preprocessing.utils.targets import TargetType

            database = LeagueDatabase()
            scores = self._scores()
            rows = []
            for league_id, config in database.index.items():
                df = database.load_league(league_id=league_id)
                league_scores = scores[scores['league_id'] == league_id]
                rows.append({
                    'league_id': league_id,
                    'country': config.league.country,
                    'name': config.league.name,
                    'category': config.league.category,
                    'matches': int(df.shape[0]),
                    'first_season': int(df['Season'].min()),
                    'last_match': str(df['Date'].max()),
                    'markets': [target.value for target in TargetType.available(df)],
                    'models': int(league_scores.shape[0]),
                })
            return sorted(rows, key=lambda row: (row['country'], row['name']))

        return self._cached('leagues', load)

    def models(self) -> List[Dict[str, Any]]:
        return _records(self._scores())

    def predictions(self) -> List[Dict[str, Any]]:
        def load():
            from src.core.automation.insights import preferred_lookup
            from src.core.automation.ledger import PredictionLedger

            since = (date.today() - timedelta(days=120)).isoformat()
            df = PredictionLedger().predictions(since=since)
            if df.empty:
                return []

            preferred = preferred_lookup(self._scores())
            df['preferred'] = [
                preferred.get((league_id, market)) == model_id
                for league_id, market, model_id in zip(df['league_id'], df['market'], df['model_id'])
            ]
            today = date.today().isoformat()
            df['status'] = [
                ('won' if correct == 1 else 'lost') if pd.notna(settled_at) else
                ('upcoming' if match_date >= today else 'awaiting result')
                for settled_at, correct, match_date in zip(df['settled_at'], df['correct'], df['match_date'])
            ]
            df['probabilities'] = df['probabilities'].map(json.loads)
            return _records(df)

        return self._cached('predictions', load)

    def overview(self) -> Dict[str, Any]:
        log_path = PROJECT_ROOT / 'storage' / 'logs' / 'automation.log'
        last_run = None
        if log_path.is_file():
            with open(log_path, encoding='utf-8', errors='replace') as log_file:
                finished = [line for line in log_file if 'Run finished' in line]
            if finished:
                last_run = finished[-1][:19]
        return {
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'last_run': last_run,
        }


class JobRunner:
    """ Runs one CLI job at a time in a subprocess and keeps its output. """

    def __init__(self):
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._state: Dict[str, Any] = {'name': None, 'running': False, 'exit_code': None, 'started': None, 'finished': None}
        self._output: List[str] = []

    def start(self, name: str, on_finish) -> Dict[str, Any]:
        with self._lock:
            if self._state['running']:
                return {'started': False, 'message': f'"{self._state["name"]}" is still running.'}

            command = [sys.executable, str(PROJECT_ROOT / 'cli.py'), *JOB_COMMANDS[name]]
            self._process = subprocess.Popen(
                command, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )
            self._output = []
            self._state = {
                'name': name, 'running': True, 'exit_code': None,
                'started': datetime.now().isoformat(timespec='seconds'), 'finished': None
            }

        def follow(process: subprocess.Popen):
            for line in process.stdout:
                line = line.rstrip()
                # Progress bars and library warnings are noise in the dashboard.
                if line and 'it/s' not in line and 'Warning' not in line:
                    with self._lock:
                        self._output = (self._output + [line])[-200:]
            exit_code = process.wait()
            with self._lock:
                self._state.update(running=False, exit_code=exit_code, finished=datetime.now().isoformat(timespec='seconds'))
            on_finish()

        threading.Thread(target=follow, args=(self._process,), daemon=True).start()
        return {'started': True}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {**self._state, 'output': self._output[-40:]}

    def stop(self):
        with self._lock:
            if self._process is not None and self._state['running']:
                self._process.terminate()


class DashboardHandler(BaseHTTPRequestHandler):
    data: DashboardData = None
    jobs: JobRunner = None

    def log_message(self, format, *args):  # Quiet server.
        pass

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = HTTPStatus.OK):
        self._send(status, json.dumps(payload).encode('utf-8'), 'application/json')

    def do_GET(self):
        route = urlparse(self.path).path
        try:
            if route in ('/', '/index.html'):
                self._send(HTTPStatus.OK, (STATIC_DIRECTORY / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            elif route == '/api/overview':
                self._json(self.data.overview())
            elif route == '/api/leagues':
                self._json(self.data.leagues())
            elif route == '/api/models':
                self._json(self.data.models())
            elif route == '/api/predictions':
                self._json(self.data.predictions())
            elif route == '/api/jobs':
                self._json(self.jobs.status())
            elif route == '/favicon.png':
                self._send(HTTPStatus.OK, (PROJECT_ROOT / 'storage/icons/menu/ball.png').read_bytes(), 'image/png')
            else:
                self._json({'error': 'not found'}, HTTPStatus.NOT_FOUND)
        except Exception as error:
            self._json({'error': str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        # A custom header cannot be sent cross-site without a CORS preflight (which is never allowed), so other
        # websites cannot trigger jobs.
        if self.headers.get('X-ProphitBet') != '1':
            self._json({'error': 'forbidden'}, HTTPStatus.FORBIDDEN)
            return

        route = urlparse(self.path).path
        name = parse_qs(urlparse(self.path).query).get('name', [''])[0]
        if route == '/api/jobs' and name in JOB_COMMANDS:
            self._json(self.jobs.start(name, on_finish=self.data.invalidate))
        else:
            self._json({'error': 'unknown job'}, HTTPStatus.BAD_REQUEST)


def serve(port: int = 8765, open_browser: bool = True):
    """ Starts the dashboard at http://127.0.0.1:<port> (blocking). """

    os.chdir(PROJECT_ROOT)
    DashboardHandler.data = DashboardData()
    DashboardHandler.jobs = JobRunner()
    server = ThreadingHTTPServer(('127.0.0.1', port), DashboardHandler)
    url = f'http://127.0.0.1:{port}/'
    print(f'ProphitBet dashboard running at {url} (Ctrl+C to stop)', flush=True)

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        DashboardHandler.jobs.stop()
        server.server_close()
