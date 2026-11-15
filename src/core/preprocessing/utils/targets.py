import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Tuple

# Raw half-time score columns. They are kept in the league dataset when the source provides them
# (football-data.co.uk "main" leagues), so that half-time markets can be derived.
HALF_TIME_COLUMNS = ['HTHG', 'HTAG']

# Optional raw columns of football-data.co.uk "main" leagues. Unlike the core columns, they may be missing for some
# seasons (e.g. corners & cards before 2016-2019 in some leagues): such matches are only excluded from the markets
# that need them.
CORNER_COLUMNS = ['HC', 'AC']                       # Corners of home / away team.
CARD_COLUMNS = ['HY', 'AY', 'HR', 'AR']             # Yellow & red cards of home / away team.
ASIAN_HANDICAP_LINE = 'AHh'                         # Main Asian handicap line (home team's handicap), pre-match.
ASIAN_HANDICAP_ODDS = ['AHOddH', 'AHOddA']          # Average odds of the home / away side of the main line.
OPTIONAL_RAW_COLUMNS = [*CORNER_COLUMNS, *CARD_COLUMNS, ASIAN_HANDICAP_LINE, *ASIAN_HANDICAP_ODDS]

# Form statistics computed from the optional raw columns (only used by the markets that need them).
CORNER_STATS = ['HCF', 'HCA', 'ACF', 'ACA']         # Recent corners for / against (home team at home, away team away).
CARD_STATS = ['HBK', 'HBKA', 'ABK', 'ABKA']         # Recent cards (yellow 1, red 2) received / by opponents.
MARKET_STATS = [*CORNER_STATS, *CARD_STATS]

# Bookings count yellow cards as 1 and red cards as 2.
RED_CARD_WEIGHT = 2


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
                8. CORNERS_*: total corners under/over 7.5 - 11.5 (requires corners).
                9. CARDS_*: total cards (yellow 1, red 2) under/over 2.5 - 5.5 (requires cards).
                10. ASIAN_HANDICAP: which side covers the bookmaker's main Asian handicap line, H/A (pushes have no label).

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
    CORNERS_7_5 = 'Corners-U/O-7.5'
    CORNERS_8_5 = 'Corners-U/O-8.5'
    CORNERS_9_5 = 'Corners-U/O-9.5'
    CORNERS_10_5 = 'Corners-U/O-10.5'
    CORNERS_11_5 = 'Corners-U/O-11.5'
    CARDS_2_5 = 'Cards-U/O-2.5'
    CARDS_3_5 = 'Cards-U/O-3.5'
    CARDS_4_5 = 'Cards-U/O-4.5'
    CARDS_5_5 = 'Cards-U/O-5.5'
    ASIAN_HANDICAP = 'AH-Main'

    def get_mapper(self) -> Dict[str, int]:
        """ Returns the label -> class index mapping of the target. """

        try:
            return dict(_TARGET_SPECS[self].mapper)
        except KeyError:
            raise NotImplementedError(f'Not supported target type: "{self.name}"')

    @property
    def required_columns(self) -> List[str]:
        """ Raw columns required to compute the target's labels. """

        return list(_TARGET_SPECS[self].required)

    @property
    def extra_features(self) -> List[str]:
        """ Additional model inputs used only by this target (e.g. corner form for corner markets). """

        return list(_TARGET_SPECS[self].extra_features)

    @property
    def optional(self) -> bool:
        """ Whether the target's labels may be missing for some matches (optional data or pushes). """

        return _TARGET_SPECS[self].optional

    def compute_labels(self, df: pd.DataFrame) -> pd.Series:
        """ Computes the target labels from the raw columns of a league dataframe (missing where unknown). """

        return _TARGET_SPECS[self].label_fn(df)

    def is_available(self, df: pd.DataFrame, strict: bool = False, min_labels: int = 200) -> bool:
        """ Whether the target can be used with the dataframe.

            Args
            ----
            :param df: (pd.DataFrame) league's dataframe.
            :param strict: (bool) require labels (and extra features) for every match, e.g. for GUI views that
                           evaluate all matches. Otherwise, optional targets need at least `min_labels` labelled matches.
            :param min_labels: (int) minimum labelled matches of optional targets (non-strict mode).
        """

        if self.value not in df.columns or not set(self.extra_features).issubset(df.columns):
            return False
        if not self.optional:
            return True
        if strict:
            return bool(df[[self.value, *self.extra_features]].notna().all().all())
        return int(df[self.value].notna().sum()) >= min_labels

    @staticmethod
    def available(df: pd.DataFrame, strict: bool = False) -> List['TargetType']:
        """ Returns the targets (markets) that can be used with the dataframe (see is_available). """

        return [target for target in TargetType if target.is_available(df, strict=strict)]

    @staticmethod
    def label_columns() -> List[str]:
        """ Returns the label column names of all targets. """

        return [target.value for target in TargetType]

    @staticmethod
    def optional_columns() -> List[str]:
        """ Columns that may be missing for some matches without excluding them from the other markets. """

        return [*OPTIONAL_RAW_COLUMNS, *MARKET_STATS, *[target.value for target in TargetType if target.optional]]


def _under_over(values: pd.Series, line: float) -> pd.Series:
    return values.gt(line).map({True: 'O', False: 'U'}).where(values.notna())


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


def total_corners(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df['HC'], errors='coerce') + pd.to_numeric(df['AC'], errors='coerce')


def total_cards(df: pd.DataFrame) -> pd.Series:
    """ Total bookings: yellow cards count 1, red cards count 2. """

    cards = df[CARD_COLUMNS].apply(pd.to_numeric, errors='coerce')
    return cards['HY'] + cards['AY'] + RED_CARD_WEIGHT*(cards['HR'] + cards['AR'])


def _asian_handicap(df: pd.DataFrame) -> pd.Series:
    """ H if the home side covers the main line (home goals + line > away goals), A if the away side covers. Pushes
        and matches without a line have no label. Quarter lines are labelled by the side that (half-)wins.
    """

    margin = df['HG'] - df['AG'] + pd.to_numeric(df[ASIAN_HANDICAP_LINE], errors='coerce')
    labels = pd.Series(np.where(margin > 0, 'H', 'A'), index=df.index, dtype=object)
    return labels.where(margin.notna() & (margin != 0))


@dataclass(frozen=True)
class _TargetSpec:
    mapper: Tuple
    required: Tuple
    label_fn: Callable[[pd.DataFrame], pd.Series]
    extra_features: Tuple = field(default=())
    optional: bool = False


_UO = (('U', 0), ('O', 1))
_NY = (('N', 0), ('Y', 1))
_HDA = (('H', 0), ('D', 1), ('A', 2))
_HA = (('H', 0), ('A', 1))
_FT = ('HG', 'AG')
_HT = ('HG', 'AG', 'HTHG', 'HTAG')


def _corners_spec(line: float) -> _TargetSpec:
    return _TargetSpec(_UO, tuple(CORNER_COLUMNS), lambda df: _under_over(total_corners(df), line), tuple(CORNER_STATS), True)


def _cards_spec(line: float) -> _TargetSpec:
    return _TargetSpec(_UO, tuple(CARD_COLUMNS), lambda df: _under_over(total_cards(df), line), tuple(CARD_STATS), True)


_TARGET_SPECS: Dict[TargetType, _TargetSpec] = {
    TargetType.RESULT: _TargetSpec(_HDA, _FT, lambda df: _result(df['HG'], df['AG'])),
    TargetType.RESULT_UNDER_OVER: _TargetSpec(_UO, _FT, lambda df: _under_over(df['HG'] + df['AG'], 2.5)),
    TargetType.UNDER_OVER_1_5: _TargetSpec(_UO, _FT, lambda df: _under_over(df['HG'] + df['AG'], 1.5)),
    TargetType.UNDER_OVER_3_5: _TargetSpec(_UO, _FT, lambda df: _under_over(df['HG'] + df['AG'], 3.5)),
    TargetType.BTTS: _TargetSpec(_NY, _FT, lambda df: _yes_no(df['HG'].gt(0) & df['AG'].gt(0))),
    TargetType.HOME_TEAM_TOTAL_0_5: _TargetSpec(_UO, _FT, lambda df: _under_over(df['HG'], 0.5)),
    TargetType.HOME_TEAM_TOTAL_1_5: _TargetSpec(_UO, _FT, lambda df: _under_over(df['HG'], 1.5)),
    TargetType.AWAY_TEAM_TOTAL_0_5: _TargetSpec(_UO, _FT, lambda df: _under_over(df['AG'], 0.5)),
    TargetType.AWAY_TEAM_TOTAL_1_5: _TargetSpec(_UO, _FT, lambda df: _under_over(df['AG'], 1.5)),
    TargetType.HALF_TIME_RESULT: _TargetSpec(_HDA, _HT, lambda df: _result(df['HTHG'], df['HTAG'])),
    TargetType.HOME_WIN_EITHER_HALF: _TargetSpec(_NY, _HT, _home_wins_either_half),
    TargetType.AWAY_WIN_EITHER_HALF: _TargetSpec(_NY, _HT, _away_wins_either_half),
    TargetType.CORNERS_7_5: _corners_spec(7.5),
    TargetType.CORNERS_8_5: _corners_spec(8.5),
    TargetType.CORNERS_9_5: _corners_spec(9.5),
    TargetType.CORNERS_10_5: _corners_spec(10.5),
    TargetType.CORNERS_11_5: _corners_spec(11.5),
    TargetType.CARDS_2_5: _cards_spec(2.5),
    TargetType.CARDS_3_5: _cards_spec(3.5),
    TargetType.CARDS_4_5: _cards_spec(4.5),
    TargetType.CARDS_5_5: _cards_spec(5.5),
    TargetType.ASIAN_HANDICAP: _TargetSpec(_HA, (*_FT, ASIAN_HANDICAP_LINE), _asian_handicap, (ASIAN_HANDICAP_LINE,), True),
}


def compute_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ Returns a dataframe with the label columns of every target whose required columns exist in df.
        Labels of optional targets are missing where their data is missing.

        Args
        ----
        :param df: (pd.DataFrame) league's dataframe with raw columns (HG, AG and optionally HTHG, HTAG, corners...).
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

    if target_type.value not in df.columns:
        raise KeyError(
            f'Target "{target_type.value}" is not available for this league. Half-time markets require leagues '
            f'with half-time scores, corner/card/Asian handicap markets require main leagues; re-download (rebuild) '
            f'leagues created before the market was added.'
        )

    mapper = target_type.get_mapper()
    labels = df[target_type.value]
    if labels.isna().any():
        raise ValueError(f'Target "{target_type.value}" has matches without labels: drop them first (drop_incomplete).')

    # Converting targets to numerical values.
    y = labels.map(mapper).to_numpy(np.int32)

    # One-hot encoding targets.
    if one_hot:
        num_classes = len(mapper)

        if num_classes < 2:
            raise ValueError(f'One-hot encoding expects num_classes to be at least 2, got {num_classes}')

        y = np.eye(num_classes, dtype=np.float32)[y]

    return y
