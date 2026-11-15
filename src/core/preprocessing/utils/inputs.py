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
