""" Upcoming fixtures (with odds) published by football-data.co.uk. Team names match its historical data.

    * fixtures.csv: "main" leagues (identified by division code, e.g. E0), including Over/Under 2.5 odds.
    * new_league_fixtures.csv: "extra" leagues (identified by country).
    The files cover the next few days only and are updated about twice a week. "explorer" leagues are not covered.
"""

import io
import re
from typing import Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# football-data.co.uk publishes kickoff times in UK time; they are converted to SAST.
SOURCE_TIMEZONE = ZoneInfo('Europe/London')
DISPLAY_TIMEZONE = ZoneInfo('Africa/Johannesburg')
DISPLAY_TIMEZONE_NAME = 'SAST'

MAIN_FIXTURES_URL = 'https://www.football-data.co.uk/fixtures.csv'
EXTRA_FIXTURES_URL = 'https://www.football-data.co.uk/new_league_fixtures.csv'

_ODDS_PREFERENCE = {
    '1': ['AvgH', 'PSH', 'B365H', 'MaxH'],
    'X': ['AvgD', 'PSD', 'B365D', 'MaxD'],
    '2': ['AvgA', 'PSA', 'B365A', 'MaxA'],
    'Over 2.5': ['Avg>2.5', 'P>2.5', 'B365>2.5', 'Max>2.5'],
    'Under 2.5': ['Avg<2.5', 'P<2.5', 'B365<2.5', 'Max<2.5'],
    'AH Line': ['AHh'],
    'AH Home': ['AvgAHH', 'PAHH', 'B365AHH', 'MaxAHH'],
    'AH Away': ['AvgAHA', 'PAHA', 'B365AHA', 'MaxAHA'],
}


class FootballDataFixtures:
    """ Downloads (once per instance) and filters the football-data.co.uk fixture files. """

    def __init__(self, timeout: float = 20.0):
        self._timeout = timeout
        self._cache: Dict[str, pd.DataFrame] = {}
        self._headers = {'User-Agent': 'Mozilla/5.0'}

    def _download(self, url: str) -> pd.DataFrame:
        if url not in self._cache:
            response = requests.get(url, timeout=self._timeout, headers=self._headers)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.content.decode('latin1')), on_bad_lines='skip')
            df.columns = [str(column).replace('ï»¿', '').replace('﻿', '').strip() for column in df.columns]
            self._cache[url] = df
        return self._cache[url]

    @staticmethod
    def division_code(league) -> Optional[str]:
        """ Returns the division code (e.g. E0) of a "main" league from its url. """

        match = re.search(r'/([A-Z0-9]+)\.csv$', league.url)
        return match.group(1) if match else None

    @staticmethod
    def _first_available(df: pd.DataFrame, columns) -> pd.Series:
        result = pd.Series(float('nan'), index=df.index)
        for column in columns:
            if column in df.columns:
                result = result.fillna(pd.to_numeric(df[column], errors='coerce'))
        return result

    def get_fixtures(self, league) -> pd.DataFrame:
        """ Returns the league's upcoming fixtures with columns: Date (YYYY-MM-DD), Time (HH:MM), Home, Away, 1, X, 2,
            Over 2.5, Under 2.5, AH Line (home team's main Asian handicap), AH Home, AH Away (missing values are NaN). Date and Time are the kickoff in SAST (a missing kickoff time
            keeps the published date). Returns an empty dataframe if the league is not covered.

            Args
            ----
            :param league: (League) league entity (from leagues.json).
        """

        columns = ['Date', 'Time', 'Home', 'Away', '1', 'X', '2', 'Over 2.5', 'Under 2.5', 'AH Line', 'AH Home', 'AH Away']

        if league.category == 'main':
            df = self._download(MAIN_FIXTURES_URL)
            df = df[df['Div'] == self.division_code(league)].rename(columns={'HomeTeam': 'Home', 'AwayTeam': 'Away'})
        elif league.category == 'extra':
            df = self._download(EXTRA_FIXTURES_URL)
            df = df[df['Country'].astype(str).str.casefold() == league.country.casefold()]
        else:
            return pd.DataFrame(columns=columns)

        if df.empty:
            return pd.DataFrame(columns=columns)

        dates = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        times = df['Time'].astype(str).str.strip() if 'Time' in df.columns else pd.Series('', index=df.index)
        kickoffs = pd.to_datetime(
            dates.dt.strftime('%Y-%m-%d') + ' ' + times, format='%Y-%m-%d %H:%M', errors='coerce'
        ).dt.tz_localize(SOURCE_TIMEZONE, ambiguous='NaT', nonexistent='NaT').dt.tz_convert(DISPLAY_TIMEZONE)

        fixtures_df = pd.DataFrame({
            'Date': kickoffs.dt.strftime('%Y-%m-%d').where(kickoffs.notna(), dates.dt.strftime('%Y-%m-%d')),
            'Time': kickoffs.dt.strftime('%H:%M'),
            'Home': df['Home'].astype(str).str.strip(),
            'Away': df['Away'].astype(str).str.strip(),
            **{odd: self._first_available(df, preference) for odd, preference in _ODDS_PREFERENCE.items()}
        })
        return fixtures_df.dropna(subset=['Date']).reset_index(drop=True)


class FootballDataFixtureScraper:
    """ Adapter exposing the football-data.co.uk fixtures through the FootyStatsScraper interface
        (scrape -> get_available_table_dates -> parse_league_table), so the GUI can use it as a fallback.

        Attributes
        ----------
        :param league: (League) league entity (from leagues.json).
    """

    def __init__(self, league, fixtures: Optional[FootballDataFixtures] = None):
        self._league = league
        self._fixtures = fixtures or FootballDataFixtures()

    def scrape(self, fixture_url: str = '') -> pd.DataFrame:
        """ Downloads the league's fixtures. The fixture url is ignored (kept for interface compatibility). """

        return self._fixtures.get_fixtures(league=self._league)

    @staticmethod
    def _dates(tree: pd.DataFrame) -> list:
        # FootyStats lists dates from newest to oldest.
        return sorted(tree['Date'].dropna().unique().tolist(), reverse=True)

    def get_available_table_dates(self, tree: pd.DataFrame) -> Optional[list]:
        """ Returns the fixture dates as FootyStats-like labels (e.g. "Nov 21"), ordered from newest to oldest. """

        dates = self._dates(tree)
        if not dates:
            return None
        return [f'{pd.Timestamp(value):%b} {pd.Timestamp(value).day}' for value in dates]

    def parse_league_table(self, tree: pd.DataFrame, table_index: int) -> Optional[pd.DataFrame]:
        """ Returns the matches of the selected date with columns: Home | Away | 1 | X | 2 | 1X | 12 | X2. """

        dates = self._dates(tree)
        if not (0 <= table_index < len(dates)):
            return None

        df = tree[tree['Date'] == dates[table_index]][['Home', 'Away', '1', 'X', '2']].reset_index(drop=True)
        # (Kickoff times are not part of FootyStats' table format used by the GUI.)
        df[['1', 'X', '2']] = df[['1', 'X', '2']].fillna(1.0)  # FootyStats' "missing odd" sentinel.

        def double_chance(first: pd.Series, second: pd.Series) -> pd.Series:
            return (1.0/(1.0/first + 1.0/second)).round(2).clip(lower=1.0)

        df['1X'] = double_chance(df['1'], df['X'])
        df['12'] = double_chance(df['1'], df['2'])
        df['X2'] = double_chance(df['X'], df['2'])
        return df
