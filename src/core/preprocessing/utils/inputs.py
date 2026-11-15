import pandas as pd
from typing import Optional, Tuple


def construct_inputs_by_match(league_df: pd.DataFrame, match_df: pd.Series) -> Optional[pd.Series]:
    """ Constructs a model input using the home team, away team and the odds for a specified match.
        If no historical match is found between home and away teams, it returns None.

        Args
        ----
        :param league_df: (pd.DataFrame) league's dataframe.
        :param match_df: (pd.Series) match's series that includes home, away teams, and 1,X,2 odds.
    """

    home, away, odd_1, odd_x, odd_2 = match_df
    mask = (league_df['Home'] == home) & (league_df['Away'] == away)
    rows = league_df[mask]

    if rows.shape[0] == 0:
        return None

    row = rows.iloc[0]
    row[['1', 'X', '2']] = [odd_1, odd_x, odd_2]

    # Validating selected row.
    if row.isna().any():
        raise ValueError('Found missing values in rows. Remove missing values before calling this method.')

    return row


def double_chance_odd(first_odd: float, second_odd: float) -> float:
    """ Computes the double chance odd (e.g. 1X) from two single odds (e.g. 1 and X). """

    return max(round(1.0/((1.0/first_odd) + (1.0/second_odd)), 2), 1.0)


def latest_season_and_week(league_df: pd.DataFrame) -> Tuple[object, int]:
    """ Returns the latest season of the league and the week following its last played week.

        Args
        ----
        :param league_df: (pd.DataFrame) league's dataframe (including rows with missing values).
    """

    dates = pd.to_datetime(league_df['Date'], errors='coerce')
    if dates.notna().any():
        latest_season = league_df.loc[dates.idxmax(), 'Season']
    else:
        latest_season = league_df['Season'].max()

    season_rows = league_df[league_df['Season'] == latest_season]
    last_week = pd.to_numeric(season_rows['Week'], errors='coerce').max()
    if pd.isna(last_week):
        raise ValueError('Could not determine the last week of the current season.')
    return latest_season, int(last_week) + 1


def prepare_prediction_row(row: pd.Series, match_date: str, season: object, week: int) -> pd.Series:
    """ Completes a row returned by "construct_inputs_by_match" for an upcoming (unplayed) match: sets its date,
        season, week, double chance odds and neutral placeholders for the outcome columns. Outcome columns are
        metadata, so placeholders never reach the model inputs.

        Args
        ----
        :param row: (pd.Series) the match row, with the upcoming match's 1, X, 2 odds already set.
        :param match_date: (str) date of the upcoming match.
        :param season: league's current season.
        :param week: upcoming match week.
    """

    row = row.copy()
    odd_1, odd_x, odd_2 = float(row['1']), float(row['X']), float(row['2'])
    row['Date'] = match_date
    row['Season'] = season
    row['Week'] = week
    row['1X'] = double_chance_odd(odd_1, odd_x)
    row['12'] = double_chance_odd(odd_1, odd_2)
    row['X2'] = double_chance_odd(odd_x, odd_2)
    row['HG'] = 0
    row['AG'] = 0
    row['Result'] = 'D'
    row['Result-U/O'] = 'U'
    return row


def construct_inputs_by_form(league_df: pd.DataFrame, matches_df: pd.DataFrame, league_config) -> pd.DataFrame:
    """ Constructs model inputs of upcoming matches from both teams' current form: each match is appended after the
        league's played matches and its statistics are computed by the same engine (and settings) used for the
        league's data. Since statistics only use previous matches, inputs are identical to how training rows were
        built. Matches of the same date are computed together; matches of later dates never see unplayed results.

        Returns a dataframe with the league's columns, aligned with matches_df's index. Rows have missing statistics
        when a team has not played enough matches in the current season (match history window).

        Args
        ----
        :param league_df: (pd.DataFrame) league's dataframe (including rows with missing values).
        :param matches_df: (pd.DataFrame) upcoming matches with columns Date (YYYY-MM-DD), Home, Away, 1, X, 2.
        :param league_config: (LeagueConfig) league's config (statistics settings).
    """

    from src.core.preprocessing.statistics import StatisticsEngine
    from src.core.preprocessing.utils.targets import HALF_TIME_COLUMNS, TargetType

    stats_columns = [column for column in league_config.stats_columns if column in league_df.columns]
    raw_columns = [column for column in league_df.columns if column not in stats_columns]
    season, week = latest_season_and_week(league_df=league_df)

    # Statistics are computed per season, so only the current season's played matches are required.
    history_df = league_df.loc[league_df['Season'] == season, raw_columns].iloc[::-1]
    history_df = history_df.sort_values(by='Date', ascending=True, kind='stable')

    outcome_placeholders = {'HG': 0, 'AG': 0, **{column: 0 for column in HALF_TIME_COLUMNS}}
    outcome_placeholders.update({target.value: next(iter(target.get_mapper())) for target in TargetType})

    rows = []
    for match_date, date_matches in matches_df.groupby('Date', sort=True):
        upcoming_df = pd.DataFrame(index=range(date_matches.shape[0]), columns=raw_columns, dtype=object)
        upcoming_df['Date'] = match_date
        upcoming_df['Season'] = season
        upcoming_df['Week'] = week
        for column in ['Home', 'Away', '1', 'X', '2']:
            upcoming_df[column] = date_matches[column].to_numpy()
        upcoming_df['1X'] = [double_chance_odd(a, b) for a, b in zip(upcoming_df['1'], upcoming_df['X'])]
        upcoming_df['12'] = [double_chance_odd(a, b) for a, b in zip(upcoming_df['1'], upcoming_df['2'])]
        upcoming_df['X2'] = [double_chance_odd(a, b) for a, b in zip(upcoming_df['X'], upcoming_df['2'])]
        for column, value in outcome_placeholders.items():
            if column in upcoming_df.columns:
                upcoming_df[column] = value

        combined_df = pd.concat(
            [history_df[history_df['Date'] < match_date], upcoming_df], axis=0, ignore_index=True
        ).astype(history_df.dtypes.to_dict())

        stats_df = StatisticsEngine(
            match_history_window=league_config.match_history_window,
            goal_diff_margin=league_config.goal_diff_margin,
            aggregation=league_config.aggregation
        ).compute_stats(df=combined_df, stats_columns=stats_columns)

        date_rows = stats_df.iloc[-date_matches.shape[0]:]
        date_rows.index = date_matches.index
        rows.append(date_rows)

    if not rows:
        return pd.DataFrame(columns=league_df.columns)
    return pd.concat(rows, axis=0).loc[matches_df.index, league_df.columns]


def construct_inputs_from_fixture(df: pd.DataFrame, fixture_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """ Constructs a model input using a fixture (multiple matches) that include home, away, 1, X, 2 odds.
        Missing values should be omitted before calling this method.

        Args
        ----
        :param df: (pd.DataFrame) league's dataframe.
        :param fixture_df: (pd.Series) fixture's dataframe that includes home, away teams, and 1,X,2 odds.
    """

    # Ensuring nan rows are dropped.
    if df.isna().any().any():
        raise ValueError('Found missing values in dataframe. Remove missing values before calling this method.')

    # Constructing inputs for each match.
    inputs = fixture_df.apply(lambda match_df: construct_inputs_by_match(league_df=df, match_df=match_df), axis=1)

    # Validating inputs.
    valid_inputs = [x for x in inputs if x is not None]

    if not valid_inputs:
        x = None
    elif len(valid_inputs) == 1:
        x = valid_inputs[0].to_frame().T.reset_index(drop=True)
    else:
        x = pd.concat(valid_inputs, axis=0, ignore_index=True)

    return x
