""" Prediction ledger: stores every automated prediction and settles it once the match result is known, so models
    can be judged (and preferred) by their real track record rather than only by their training hold-out score.

    Stored in storage/predictions/ledger.sqlite (SQLite, standard library).
"""

import glob
import json
import math
import os
import sqlite3
import threading
from contextlib import closing
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.preprocessing.utils.targets import TargetType

DEFAULT_LEDGER_PATH = 'storage/predictions/ledger.sqlite'

# Settled predictions a model needs (for a league & market) before its live record is used to prefer it.
MIN_SETTLED_FOR_LIVE_RANKING = 30

# A fixture is matched to a played match of the same teams within this many days (postponements, time zones).
MATCH_DATE_TOLERANCE_DAYS = 3

def asian_handicap_return(line: float, goal_difference: float, pick: str, odd: Optional[float]) -> float:
    """ Return (profit per unit stake) of an Asian handicap bet. The line is the home team's handicap and the goal
        difference is home - away goals. Quarter lines (e.g. -0.75) split the stake over the two nearest half lines,
        so they can be half won / half lost; whole lines can push (stake returned). Without odds, a win counts 1.

        Args
        ----
        :param line: (float) home team's handicap.
        :param goal_difference: (float) home goals - away goals.
        :param pick: (str) "H" (home side) or "A" (away side).
        :param odd: (float | None) decimal odd of the picked side.
    """

    win = (odd - 1.0) if odd else 1.0
    quarter = abs(line*4) % 2 == 1
    lines = [line - 0.25, line + 0.25] if quarter else [line]
    total = 0.0
    for part in lines:
        margin = goal_difference + part if pick == 'H' else -(goal_difference + part)
        total += win if margin > 0 else (0.0 if margin == 0 else -1.0)
    return total/len(lines)


# Double chance predictions (derived from Result models) are settled from the full-time result.
DOUBLE_CHANCE_OUTCOMES = {'1X': ('H', 'D'), '12': ('H', 'A'), 'X2': ('D', 'A')}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league_id TEXT NOT NULL,
    match_date TEXT NOT NULL,
    kickoff_time TEXT,
    line REAL,
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    market TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL,
    probabilities TEXT NOT NULL,
    odd_1 REAL, odd_x REAL, odd_2 REAL, odd_over25 REAL, odd_under25 REAL,
    odd REAL,
    ev REAL,
    holdout_log_loss REAL,
    skill_vs_bookmaker REAL,
    inputs TEXT,
    source TEXT,
    predicted_at TEXT NOT NULL,
    actual TEXT,
    correct INTEGER,
    log_loss REAL,
    profit REAL,
    settled_at TEXT,
    UNIQUE (league_id, match_date, home, away, model_id, market)
);
CREATE INDEX IF NOT EXISTS idx_predictions_league ON predictions (league_id, settled_at);
"""


def _float(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


_SCHEMA_STATEMENTS = [statement.strip() for statement in _SCHEMA.split(';') if statement.strip()]
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY = set()


class PredictionLedger:
    """ SQLite-backed store of predictions and their outcomes.

        Attributes
        ----------
        :param filepath: (str) ledger database path.
    """

    def __init__(self, filepath: str = DEFAULT_LEDGER_PATH):
        self._filepath = filepath
        is_new = not os.path.exists(filepath)
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self._ensure_schema()

        # Importing predictions exported before the ledger existed.
        if is_new:
            self.import_csv_files(directory=os.path.dirname(filepath) or '.')

    def _ensure_schema(self):
        """ Creates or upgrades the ledger's schema exactly once per process. The checks and upgrades run inside one
            exclusive transaction, so concurrent users (dashboard requests, a scheduled run) cannot apply the same
            upgrade twice (e.g. "duplicate column name"), and a failed upgrade leaves the ledger unchanged.
        """

        path = os.path.abspath(self._filepath)
        with _SCHEMA_LOCK:
            if path in _SCHEMA_READY:
                return

            connection = sqlite3.connect(self._filepath, timeout=30, isolation_level=None)
            connection.row_factory = sqlite3.Row
            try:
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('BEGIN IMMEDIATE')
                for statement in _SCHEMA_STATEMENTS:
                    connection.execute(statement)

                # Upgrading ledgers created before kickoff times (SAST) and Asian handicap lines were stored.
                columns = {row['name'] for row in connection.execute('PRAGMA table_info(predictions)')}
                for column, column_type in (('kickoff_time', 'TEXT'), ('line', 'REAL')):
                    if column not in columns:
                        connection.execute(f'ALTER TABLE predictions ADD COLUMN {column} {column_type}')

                # Upgrading ledgers whose predictions were unique per model only: a Result model also stores double
                # chance predictions, so the market is part of a prediction's identity.
                table_sql = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'predictions'"
                ).fetchone()['sql']
                if 'model_id, market)' not in table_sql:
                    column_list = ', '.join(row['name'] for row in connection.execute('PRAGMA table_info(predictions)'))
                    connection.execute('DROP INDEX IF EXISTS idx_predictions_league')
                    connection.execute('ALTER TABLE predictions RENAME TO predictions_old')
                    for statement in _SCHEMA_STATEMENTS:
                        connection.execute(statement)
                    connection.execute(f'INSERT INTO predictions ({column_list}) SELECT {column_list} FROM predictions_old')
                    connection.execute('DROP TABLE predictions_old')

                connection.execute('COMMIT')
            except Exception:
                connection.execute('ROLLBACK')
                raise
            finally:
                connection.close()
            _SCHEMA_READY.add(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._filepath, timeout=30)
        connection.execute('PRAGMA journal_mode=WAL')
        connection.row_factory = sqlite3.Row
        return connection

    # ------------------------------------------------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------------------------------------------------

    def record(self, predictions_df: pd.DataFrame) -> int:
        """ Stores predictions (rows of an automation predictions dataframe). A prediction of the same match & model
            replaces the stored one only while it is unsettled. Returns the number of stored rows.
        """

        if predictions_df is None or predictions_df.empty:
            return 0

        now = datetime.now().isoformat(timespec='seconds')
        probability_columns = [column for column in predictions_df.columns if column.startswith('Prob-')]
        rows = []
        for _, row in predictions_df.iterrows():
            probabilities = {
                column[len('Prob-'):]: _float(row[column]) for column in probability_columns if _float(row[column]) is not None
            }
            kickoff_time = row.get('Time')
            rows.append((
                row['League'], str(row['Date']), None if pd.isna(kickoff_time) else str(kickoff_time), _float(row.get('Line')),
                row['Home'], row['Away'], row['Market'], row['Model'],
                str(row['Prediction']), _float(row.get('Confidence')), json.dumps(probabilities),
                _float(row.get('1')), _float(row.get('X')), _float(row.get('2')),
                _float(row.get('Over 2.5')), _float(row.get('Under 2.5')),
                _float(row.get('Odd')), _float(row.get('EV')),
                _float(row.get('Holdout Log Loss')), _float(row.get('Skill vs Bookmaker')),
                None if pd.isna(row.get('Inputs')) else str(row.get('Inputs')),
                None if pd.isna(row.get('Source')) else str(row.get('Source')),
                now
            ))

        with closing(self._connect()) as connection, connection:
            connection.executemany("""
                INSERT INTO predictions (
                    league_id, match_date, kickoff_time, line, home, away, market, model_id, prediction, confidence,
                    probabilities, odd_1, odd_x, odd_2, odd_over25, odd_under25, odd, ev, holdout_log_loss,
                    skill_vs_bookmaker, inputs, source, predicted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (league_id, match_date, home, away, model_id, market) DO UPDATE SET
                    kickoff_time = COALESCE(excluded.kickoff_time, predictions.kickoff_time), line = excluded.line, prediction = excluded.prediction, confidence = excluded.confidence,
                    probabilities = excluded.probabilities, odd_1 = excluded.odd_1, odd_x = excluded.odd_x,
                    odd_2 = excluded.odd_2, odd_over25 = excluded.odd_over25, odd_under25 = excluded.odd_under25,
                    odd = excluded.odd, ev = excluded.ev, holdout_log_loss = excluded.holdout_log_loss,
                    skill_vs_bookmaker = excluded.skill_vs_bookmaker, inputs = excluded.inputs,
                    source = excluded.source, predicted_at = excluded.predicted_at
                WHERE predictions.settled_at IS NULL
            """, rows)
        return len(rows)

    def import_csv_files(self, directory: str) -> int:
        """ Imports prediction CSV exports (storage/predictions/<date>/<league>.csv) into the ledger. """

        imported = 0
        for filepath in sorted(glob.glob(os.path.join(directory, '*', '*.csv'))):
            try:
                df = pd.read_csv(filepath)
                if {'League', 'Date', 'Home', 'Away', 'Market', 'Model', 'Prediction'}.issubset(df.columns):
                    imported += self.record(df)
            except Exception:
                continue
        return imported

    # ------------------------------------------------------------------------------------------------------------
    # Settling
    # ------------------------------------------------------------------------------------------------------------

    def settle(self, league_id: str, league_df: pd.DataFrame, today: Optional[date] = None) -> int:
        """ Settles the league's unsettled predictions whose match has been played (found in the league data).
            Returns the number of settled predictions.
        """

        today = today or date.today()
        with closing(self._connect()) as connection:
            pending = pd.read_sql_query(
                'SELECT * FROM predictions WHERE league_id = ? AND settled_at IS NULL AND match_date <= ?',
                connection, params=(league_id, today.isoformat())
            )
        if pending.empty:
            return 0

        played_df = league_df.dropna(subset=['HG', 'AG']).copy()
        played_df['_date'] = pd.to_datetime(played_df['Date'], errors='coerce')
        by_teams = {key: group for key, group in played_df.groupby(['Home', 'Away'], sort=False)}
        tolerance = timedelta(days=MATCH_DATE_TOLERANCE_DAYS)

        updates = []
        now = datetime.now().isoformat(timespec='seconds')
        for _, prediction in pending.iterrows():
            candidates = by_teams.get((prediction['home'], prediction['away']))
            if candidates is None:
                continue

            match_date = pd.Timestamp(prediction['match_date'])
            distances = (candidates['_date'] - match_date).abs()
            if distances.min() > tolerance:
                continue

            match = candidates.loc[distances.idxmin()]
            probabilities = json.loads(prediction['probabilities'])

            if prediction['market'] == 'AH-Main':
                # Settled at the line of the prediction (the stored historical line may be the closing one).
                line = _float(prediction['line'])
                if line is None:
                    continue
                goal_difference = float(match['HG']) - float(match['AG'])
                outcome = asian_handicap_return(line=line, goal_difference=goal_difference, pick=prediction['prediction'], odd=None)
                actual = 'H' if goal_difference + line > 0 else ('A' if goal_difference + line < 0 else 'P')
                odd = _float(prediction['odd'])
                profit = None if odd is None else asian_handicap_return(line, goal_difference, prediction['prediction'], odd)
                pick_probability = min(max(float(probabilities.get(prediction['prediction'], 0.0)), 1e-15), 1.0 - 1e-15)
                if outcome > 0:
                    correct, log_loss = 1, -math.log(pick_probability)
                elif outcome < 0:
                    correct, log_loss = 0, -math.log(1.0 - pick_probability)
                else:
                    correct, log_loss = None, None  # Push: stake returned, not scored.
                updates.append((actual, correct, log_loss, profit, now, int(prediction['id'])))
                continue

            if prediction['market'] == 'Double-Chance':
                # Overlapping options: the log loss scores the picked option (binary).
                if pd.isna(match.get('Result')) or prediction['prediction'] not in DOUBLE_CHANCE_OUTCOMES:
                    continue
                actual = str(match['Result'])
                correct = int(actual in DOUBLE_CHANCE_OUTCOMES[prediction['prediction']])
                pick_probability = min(max(float(probabilities.get(prediction['prediction'], 0.0)), 1e-15), 1.0 - 1e-15)
                log_loss = -math.log(pick_probability if correct else 1.0 - pick_probability)
            else:
                if prediction['market'] not in match.index or pd.isna(match[prediction['market']]):
                    continue
                actual = str(match[prediction['market']])
                correct = int(actual == prediction['prediction'])
                log_loss = -math.log(min(max(float(probabilities.get(actual, 0.0)), 1e-15), 1.0))

            odd = _float(prediction['odd'])
            profit = None if odd is None else (odd - 1.0 if correct else -1.0)
            updates.append((actual, correct, log_loss, profit, now, int(prediction['id'])))

        if updates:
            with closing(self._connect()) as connection, connection:
                connection.executemany(
                    'UPDATE predictions SET actual = ?, correct = ?, log_loss = ?, profit = ?, settled_at = ? WHERE id = ?',
                    updates
                )
        return len(updates)

    # ------------------------------------------------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------------------------------------------------

    def predictions(self, settled: Optional[bool] = None, since: Optional[str] = None) -> pd.DataFrame:
        """ Returns stored predictions (all, only settled, or only unsettled), optionally from a match date on. """

        query = 'SELECT * FROM predictions WHERE 1 = 1'
        params: List = []
        if settled is True:
            query += ' AND settled_at IS NOT NULL'
        elif settled is False:
            query += ' AND settled_at IS NULL'
        if since:
            query += ' AND match_date >= ?'
            params.append(since)
        query += ' ORDER BY match_date, kickoff_time, league_id, home, market, model_id'

        with closing(self._connect()) as connection:
            return pd.read_sql_query(query, connection, params=params)

    def model_records(self) -> pd.DataFrame:
        """ Live track record per league, market and model (settled predictions only). """

        settled_df = self.predictions(settled=True)
        columns = ['league_id', 'market', 'model_id', 'settled', 'accuracy', 'log_loss', 'bets', 'profit', 'roi']
        if settled_df.empty:
            return pd.DataFrame(columns=columns)

        def summarize(group: pd.DataFrame) -> pd.Series:
            bets = group['profit'].notna().sum()
            profit = group['profit'].sum() if bets else np.nan
            return pd.Series({
                'settled': len(group),
                'accuracy': group['correct'].mean(),
                'log_loss': group['log_loss'].mean(),
                'bets': bets,
                'profit': profit,
                'roi': profit/bets if bets else np.nan
            })

        records = settled_df.groupby(['league_id', 'market', 'model_id']).apply(summarize, include_groups=False)
        return records.reset_index()[columns]


def preferred_models(model_scores: pd.DataFrame, min_settled: int = MIN_SETTLED_FOR_LIVE_RANKING) -> Dict:
    """ Picks the preferred model of every league & market.

        model_scores columns: league_id, market, model_id, holdout_log_loss, settled, log_loss (live).
        When every model of a league & market has at least `min_settled` settled predictions, the one with the lowest
        live log loss is preferred ("live record"); otherwise the one with the lowest hold-out log loss from training.

        Returns {(league_id, market): (model_id, basis)}.
    """

    preferred = {}
    for (league_id, market), group in model_scores.groupby(['league_id', 'market']):
        settled = group['settled'].fillna(0)
        if (settled >= min_settled).all() and group['log_loss'].notna().all():
            best = group.loc[group['log_loss'].idxmin()]
            preferred[(league_id, market)] = (best['model_id'], 'live record')
        elif group['holdout_log_loss'].notna().any():
            best = group.loc[group['holdout_log_loss'].idxmin()]
            preferred[(league_id, market)] = (best['model_id'], 'hold-out')
    return preferred
