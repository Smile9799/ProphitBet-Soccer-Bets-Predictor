import numpy as np
import pandas as pd
from enum import Enum
from typing import Callable, Dict, List, Tuple

# Raw half-time score columns. They are kept in the league dataset when the source provides them
# (football-data.co.uk "main" leagues), so that half-time markets can be derived.
HALF_TIME_COLUMNS = ['HTHG', 'HTAG']


class TargetType(Enum):
    """ Supported target types (betting markets) that describe the classification tasks.

        Each target's value is also the name of the dataset column holding its labels.

        Supported Types:
                1. RESULT: full-time result H (Home Win), D (Draw) or A (Away Win).
                2. RESULT_UNDER_OVER: under/over 2.5 total goals.
                3. UNDER_OVER_1_5 / UNDER_OVER_3_5: under/over 1.5 and 3.5 total goals.
                4. BTTS: both teams to score (N/Y).
                5. HOME/AWAY_TEAM_TOTAL_0_5/1_5: team total goals under/over 0.5 and 1.5.
                6. HALF_TIME_RESULT: half-time result H/D/A (requires half-time scores).
                7. HOME/AWAY_WIN_EITHER_HALF: team wins at least one half (N/Y) (requires half-time scores).

        NOTE: The value of existing members must never change, as stored models pickle targets by value.
    """

    RESULT = 'Result'
    RESULT_UNDER_OVER = 'Result-U/O'
    UNDER_OVER_1_5 = 'U/O-1.5'
    UNDER_OVER_3_5 = 'U/O-3.5'
    BTTS = 'BTTS'
    HOME_TEAM_TOTAL_0_5 = 'Home-U/O-0.5'
    HOME_TEAM_TOTAL_1_5 = 'Home-U/O-1.5'
    AWAY_TEAM_TOTAL_0_5 = 'Away-U/O-0.5'
    AWAY_TEAM_TOTAL_1_5 = 'Away-U/O-1.5'
    HALF_TIME_RESULT = 'HT-Result'
    HOME_WIN_EITHER_HALF = 'Home-Win-Either-Half'
    AWAY_WIN_EITHER_HALF = 'Away-Win-Either-Half'

    def get_mapper(self) -> Dict[str, int]:
        """ Returns the label -> class index mapping of the target. """

        try:
            return dict(_TARGET_SPECS[self][0])
        except KeyError:
            raise NotImplementedError(f'Not supported target type: "{self.name}"')

    @property
    def required_columns(self) -> List[str]:
        """ Raw score columns required to compute the target's labels. """

        return list(_TARGET_SPECS[self][1])

    def compute_labels(self, df: pd.DataFrame) -> pd.Series:
        """ Computes the target labels from the raw score columns of a league dataframe. """

        return _TARGET_SPECS[self][2](df)

    def is_available(self, df: pd.DataFrame) -> bool:
        """ Whether the target's label column exists in the provided dataframe. """

        return self.value in df.columns

    @staticmethod
    def available(df: pd.DataFrame) -> List['TargetType']:
        """ Returns the targets (markets) whose label columns exist in the provided dataframe. """

        return [target for target in TargetType if target.is_available(df)]

    @staticmethod
    def label_columns() -> List[str]:
        """ Returns the label column names of all targets. """

        return [target.value for target in TargetType]


def _under_over(goals: pd.Series, line: float) -> pd.Series:
    return goals.gt(line).map({True: 'O', False: 'U'})


def _yes_no(mask: pd.Series) -> pd.Series:
    return mask.map({True: 'Y', False: 'N'})


def _result(home_goals: pd.Series, away_goals: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([home_goals > away_goals, home_goals == away_goals], ['H', 'D'], default='A'),
        index=home_goals.index
    )


def _home_wins_either_half(df: pd.DataFrame) -> pd.Series:
    first_half = df['HTHG'] > df['HTAG']
    second_half = (df['HG'] - df['HTHG']) > (df['AG'] - df['HTAG'])
    return _yes_no(first_half | second_half)


def _away_wins_either_half(df: pd.DataFrame) -> pd.Series:
    first_half = df['HTAG'] > df['HTHG']
    second_half = (df['AG'] - df['HTAG']) > (df['HG'] - df['HTHG'])
    return _yes_no(first_half | second_half)


_UO = (('U', 0), ('O', 1))
_NY = (('N', 0), ('Y', 1))
_HDA = (('H', 0), ('D', 1), ('A', 2))
_FT = ('HG', 'AG')
_HT = ('HG', 'AG', 'HTHG', 'HTAG')

# target -> (label mapper, required columns, label function)
_TARGET_SPECS: Dict[TargetType, Tuple[Tuple, Tuple, Callable[[pd.DataFrame], pd.Series]]] = {
    TargetType.RESULT: (_HDA, _FT, lambda df: _result(df['HG'], df['AG'])),
    TargetType.RESULT_UNDER_OVER: (_UO, _FT, lambda df: _under_over(df['HG'] + df['AG'], 2.5)),
    TargetType.UNDER_OVER_1_5: (_UO, _FT, lambda df: _under_over(df['HG'] + df['AG'], 1.5)),
    TargetType.UNDER_OVER_3_5: (_UO, _FT, lambda df: _under_over(df['HG'] + df['AG'], 3.5)),
    TargetType.BTTS: (_NY, _FT, lambda df: _yes_no(df['HG'].gt(0) & df['AG'].gt(0))),
    TargetType.HOME_TEAM_TOTAL_0_5: (_UO, _FT, lambda df: _under_over(df['HG'], 0.5)),
    TargetType.HOME_TEAM_TOTAL_1_5: (_UO, _FT, lambda df: _under_over(df['HG'], 1.5)),
    TargetType.AWAY_TEAM_TOTAL_0_5: (_UO, _FT, lambda df: _under_over(df['AG'], 0.5)),
    TargetType.AWAY_TEAM_TOTAL_1_5: (_UO, _FT, lambda df: _under_over(df['AG'], 1.5)),
    TargetType.HALF_TIME_RESULT: (_HDA, _HT, lambda df: _result(df['HTHG'], df['HTAG'])),
    TargetType.HOME_WIN_EITHER_HALF: (_NY, _HT, _home_wins_either_half),
    TargetType.AWAY_WIN_EITHER_HALF: (_NY, _HT, _away_wins_either_half),
}


def compute_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ Returns a dataframe with the label columns of every target whose required score columns exist in df.
        Rows must not contain missing scores in the required columns.

        Args
        ----
        :param df: (pd.DataFrame) league's dataframe with raw score columns (HG, AG and optionally HTHG, HTAG).
    """

    columns = {
        target.value: target.compute_labels(df)
        for target in TargetType
        if set(target.required_columns).issubset(df.columns)
    }
    return pd.DataFrame(columns, index=df.index)


def construct_targets(df: pd.DataFrame, target_type: TargetType, one_hot: bool = False) -> np.ndarray:
    """ Constructs dataset targets for the selected classification task.
         Returns a tuple of inputs (np.ndarray), normalizer (TransformerMixin)

        Args
        ----
        :param df: (pd.DataFrame) league's dataframe.
        :param target_type: (TargetType) target type that describes the classification tasks.
        :param one_hot: (bool) whether to one-hot encode the targets. It is applied if number of targets > 2.
    """

    if not target_type.is_available(df):
        raise KeyError(
            f'Target "{target_type.value}" is not available for this league. Half-time markets require leagues '
            f'with half-time scores; re-download (rebuild) leagues created before the market was added.'
        )

    mapper = target_type.get_mapper()

    # Converting targets to numerical values.
    y = df[target_type.value].map(mapper).to_numpy(np.int32)

    # One-hot encoding targets.
    if one_hot:
        num_classes = len(mapper)

        if num_classes < 2:
            raise ValueError(f'One-hot encoding expects num_classes to be at least 2, got {num_classes}')

        y = np.eye(num_classes, dtype=np.float32)[y]

    return y
