import unicodedata
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from src.core.leagues.league import LeagueConfig
from src.core.preprocessing.utils.targets import HALF_TIME_COLUMNS, OPTIONAL_RAW_COLUMNS, TargetType, compute_target_columns


class LeagueDownloader(ABC):
    """ Base class that downloads and processes league data from open-source webpages.
        Each league requires the specified columns:
            * Date: date of match.
            * Season: season year of match.
            * Home: name of home team.
            * Away: name of away team.
            * HG: goals scored by home team.
            * AG: goals scored by away team.
            * Result: match result (H/D/A) corresponding to home, draw, or away win.
            * 1: odd for home win.
            * X: odd for draw.
            * 2: odd for away win.
    """

    def __init__(self):
        self._expected_columns = ['Date', 'Season', 'Home', 'Away', 'HG', 'AG', 'Result', '1', 'X', '2']
        self._relevant_columns = [
            'Date', 'Season', 'Week', 'Home', 'Away', 'HG', 'AG', 'Result',
            'Result-U/O', '1', 'X', '2', '1X', '12', 'X2'
        ]
        self._time_format = '%Y-%m-%d'

        # Half-time scores are kept only if (nearly) all matches provide them. Matches missing them are dropped,
        # otherwise the columns are discarded and half-time markets are unavailable for the league.
        self._max_missing_half_time_ratio = 0.02

    @abstractmethod
    def _download_dataframe(self, league_config: LeagueConfig) -> pd.DataFrame:
        """ Downloads and returns league's dataframe based on the selected parameters.

            Args
            ----
            :param league_config: (LeagueConfig) league's config provided by the user.
        """

        pass

    def _preprocess_dataframe(self, df: pd.DataFrame, league_config: LeagueConfig) -> pd.DataFrame:
        """ Preprocesses the downloaded dataframe as follows:

            Steps:
                1. Drops matches with NULL dates or results.
                2. Removes non-ASCII characters.
                3. Filters league's start year & reorders columns.
                4. Sorts matches by date in ascending order.
                5. Drops duplicated matches.
                6. Computes Result-U/O column (useful for U/O Classification tasks).
                7. Fills missing odds based on match history.
                8. Computes double chance odds: 1X, 12, X2.
                9. Computes match week.
                10. Computes the labels of the additional markets.
                11. Filters league's relevant columns.
            Args
            ----
            :param df: (pd.DataFrame) Downloaded league's data.
            :param league_config: (LeagueConfig) league's config provided by the user.
         """

        # 1. Dropping matches with missing Dates or Results.
        df = df.dropna(subset=['Date', 'Result'])

        # 1b. Keeping half-time scores (if provided) to support half-time markets.
        has_half_time = False
        if set(HALF_TIME_COLUMNS).issubset(df.columns):
            df[HALF_TIME_COLUMNS] = df[HALF_TIME_COLUMNS].apply(pd.to_numeric, errors='coerce')
            missing_half_time = df[HALF_TIME_COLUMNS].isna().any(axis=1)

            if df.shape[0] > 0 and missing_half_time.mean() <= self._max_missing_half_time_ratio:
                df = df[~missing_half_time]
                has_half_time = True

        # 1c. Keeping optional data (corners, cards, Asian handicap) when provided. Missing values are allowed: such
        #     matches are only excluded from the markets that need the data.
        optional_columns = [column for column in OPTIONAL_RAW_COLUMNS if column in df.columns]
        if optional_columns:
            df[optional_columns] = df[optional_columns].apply(pd.to_numeric, errors='coerce')
            optional_columns = [column for column in optional_columns if df[column].notna().any()]

        # 2. Removing non-ASCII characters.
        def remove_non_ascii(x: str) -> str:
            """ Removes non-ascii characters. """

            return unicodedata.normalize('NFKD', x).encode('ascii', errors='ignore').decode('ascii')

        df['Home'] = df['Home'].apply(remove_non_ascii)
        df['Away'] = df['Away'].apply(remove_non_ascii)

        # 3. Filtering start year.
        df = df[df['Season'] >= league_config.start_year]

        # 4. Sorting matches by date in ascending order.
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, utc=True, format='mixed').dt.strftime(self._time_format)
        df = df.sort_values(by='Date', ascending=True)

        # 5. Dropping duplicated matches. Index is reset, as no further matches are dropped.
        df = df.drop_duplicates(subset=['Season', 'Date', 'Home', 'Away'], ignore_index=True)

        # 6. Computing Result-U/0 column.
        results_under_over = (df['HG'] + df['AG']).ge(2.5).map({True: 'O', False: 'U'})

        # 7. Fill missing odds between same matches based on match history.
        grouped = df.groupby(by=['Home', 'Away'], sort=False, group_keys=False)[['1', 'X', '2']]
        df[['1', 'X', '2']] = grouped.bfill().fillna(grouped.ffill())

        # 8. Computing double chance odds: 1X, 12, X2
        odd_1_prob = 1.0/df['1']
        odd_x_prob = 1.0/df['X']
        odd_2_prob = 1.0/df['2']
        odds_1x = (1.0/(odd_1_prob + odd_x_prob)).round(decimals=2)
        odds_12 = (1.0/(odd_1_prob + odd_2_prob)).round(decimals=2)
        odds_x2 = (1.0/(odd_x_prob + odd_2_prob)).round(decimals=2)

        # 9. Computing match week.
        long_df = pd.DataFrame({
            'Season': np.repeat(df['Season'].to_numpy(), 2),
            'Team': np.column_stack([
                df['Home'].to_numpy(),
                df['Away'].to_numpy()
            ]).ravel()
        })
        week_per_team = long_df.groupby(['Season', 'Team'], sort=False).cumcount().add(1).to_numpy()
        week_values = week_per_team.reshape(df.shape[0], 2).max(axis=1)

        # 10. Computing the labels of the additional markets (BTTS, U/O lines, team totals, half-time markets).
        market_df = compute_target_columns(df=df).drop(
            columns=[TargetType.RESULT.value, TargetType.RESULT_UNDER_OVER.value],
            errors='ignore'
        )
        if not has_half_time:
            market_df = market_df.drop(
                columns=[target.value for target in TargetType if 'HTHG' in target.required_columns],
                errors='ignore'
            )

        # 11. Filtering relevant columns & re-ordering of columns.
        computed_df = pd.DataFrame({
            'Result-U/O': results_under_over,
            '1X': odds_1x.clip(lower=1.0),
            '12': odds_12.clip(lower=1.0),
            'X2': odds_x2.clip(lower=1.0),
            'Week': week_values
        })
        df = pd.concat([df, computed_df, market_df], axis=1)

        relevant_columns = list(self._relevant_columns)
        if has_half_time:
            ag_index = relevant_columns.index('AG')
            relevant_columns[ag_index + 1: ag_index + 1] = HALF_TIME_COLUMNS
            df[HALF_TIME_COLUMNS] = df[HALF_TIME_COLUMNS].astype(int)
        uo_index = relevant_columns.index('Result-U/O')
        relevant_columns[uo_index + 1: uo_index + 1] = market_df.columns.tolist()
        relevant_columns += optional_columns

        df = df[relevant_columns]
        return df

    def download(self, league_config: LeagueConfig) -> pd.DataFrame:
        """ Downloads, validates and preprocesses the dataframe with the requested league data.

            Args
            ----
            :param league_config: (LeagueConfig) league's config provided by the user.
        """

        # Downloading dataframe.
        df = self._download_dataframe(league_config=league_config)

        # Validating dataframe: checking for empty rows and missing columns.
        if df is None or df.shape[0] == 0:
            raise ValueError(f'Download failed: no matches were found for league "{league_config.league}".')
        if not set(self._expected_columns).issubset(df.columns):
            raise KeyError(
                f'Download failed: expected dataframe columns: "{self._expected_columns}", '
                f'found missing: "{list(set(self._expected_columns).difference(df.columns))}"'
            )

        # Preprocessing dataframe.
        df = self._preprocess_dataframe(df=df, league_config=league_config)
        return df
