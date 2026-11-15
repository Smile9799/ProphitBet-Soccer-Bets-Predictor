import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
from imblearn.base import BaseSampler
from sklearn.base import TransformerMixin
from src.core.preprocessing.utils import (
    NormalizerType,
    normalize_inputs,
    SamplerType,
    sample_inputs,
    TargetType,
    construct_targets
)
from src.core.preprocessing.utils.targets import HALF_TIME_COLUMNS, MARKET_STATS, OPTIONAL_RAW_COLUMNS


def core_columns(df: pd.DataFrame) -> List[str]:
    """ Columns that must be present (non-missing) for a match to be used at all. Optional columns (corners, cards,
        Asian handicap and their markets) may be missing for some matches.
    """

    optional = set(TargetType.optional_columns())
    return [column for column in df.columns if column not in optional]


def has_missing_core(df: pd.DataFrame) -> bool:
    """ Whether any match has missing values in the core columns. """

    return bool(df[core_columns(df)].isna().any().any())


def drop_incomplete(df: pd.DataFrame, target_type: Optional[TargetType] = None, ignore_index: bool = False) -> pd.DataFrame:
    """ Drops matches with missing core values (e.g. early-season matches without statistics). With a target, also
        drops matches without that target's label or extra inputs (e.g. seasons without corner data for corner markets).
        Use this instead of df.dropna(), which would also drop matches that only lack optional data.

        Args
        ----
        :param df: (pd.DataFrame) league's dataframe.
        :param target_type: (TargetType | None) target (market) the matches are used for.
        :param ignore_index: (bool) whether to reset the index.
    """

    subset = core_columns(df)
    if target_type is not None:
        subset += [column for column in [target_type.value, *target_type.extra_features] if column in df.columns]
    return df.dropna(subset=subset, ignore_index=ignore_index)


class DatasetPreprocessor:
    """ Multipurpose dataset processing class.

        Purposes:
            1. Dropping rows with missing values.
            2. Normalizes inputs, constructs and encodes targets.
     """

    def __init__(self):
        # Every column that is not a model input. Match outcomes (scores, corners, cards & market labels) must always be
        # listed here, otherwise they would leak into the model inputs. Market-specific inputs (corner/card form,
        # Asian handicap line) are metadata too and are only added for the markets that use them.
        self._metadata_columns = [
            'Date', 'Season', 'Week', 'Home', 'Away', '1X', '12', 'X2', 'HG', 'AG', *HALF_TIME_COLUMNS,
            *OPTIONAL_RAW_COLUMNS, *MARKET_STATS, *TargetType.label_columns()
        ]

    @property
    def metadata_columns(self) -> List[str]:
        return self._metadata_columns

    def get_metadata_columns(self, df: pd.DataFrame) -> List[str]:
        """ Returns the metadata columns that exist in the provided dataframe (markets vary per league). """

        return [column for column in self._metadata_columns if column in df.columns]

    def feature_names(self, df: pd.DataFrame, target_type: Optional[TargetType] = None) -> List[str]:
        """ Returns the model input columns: the league's statistics & odds, plus the target's extra inputs. """

        base = df.columns.drop(self.get_metadata_columns(df=df)).tolist()
        extra = [] if target_type is None else [column for column in target_type.extra_features if column not in base]
        return base + extra

    def construct_inputs(
            self,
            df: pd.DataFrame,
            normalizer: Optional[Union[NormalizerType, TransformerMixin]] = None,
            target_type: Optional[TargetType] = None
    ) -> Tuple[np.ndarray, Optional[TransformerMixin]]:
        """ Construct dataset inputs. The same pipeline is applied for every classification task; markets with extra
            inputs (e.g. corner form for corner markets) get those appended.

            Args
            ----
            :param df: (pd.DataFrame) league's dataframe.
            :param normalizer: (Normalizer) normalization method of the input data. Defaults to None.
            :param target_type: (TargetType | None) target of the model, which determines its extra inputs.
        """

        # Construct inputs.
        x = df[self.feature_names(df=df, target_type=target_type)].to_numpy(dtype=np.float32)

        # Normalizing inputs.
        if normalizer is not None:
            x, normalizer = normalize_inputs(x=x, normalizer=normalizer)

        return x, normalizer

    def preprocess(
            self,
            df: pd.DataFrame,
            target_type: TargetType,
            one_hot_targets: bool = False,
            normalizer: Optional[Union[NormalizerType, TransformerMixin]] = None,
            sampler: Optional[Union[SamplerType, BaseSampler]] = None,
            seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[TransformerMixin]]:
        """ Preprocesses the dataframe and returns ready-to-train dataset, consisting of (input, target) pairs.
            Returns a tuple of: inputs (np.ndarray), targets (np.ndarray), normalizer (TransformerMixin)

            Args
            ----
            :param df: (pd.DataFrame) league's dataframe.
            :param target_type: (TargetType) specified target type that describes the classification task.
            :param one_hot_targets: (bool) whether to one-hot encode the target data.
            :param normalizer: (Normalizer) normalization method of the input data. Defaults to None.
            :param sampler: (Sampler) sampling method of the input data. Defaults to None
            :param seed: random seed of the sampler. Defaults to None.
            :return:
        """

        used_columns = [*self.feature_names(df=df, target_type=target_type), target_type.value]
        if df[[column for column in used_columns if column in df.columns]].isna().any().any():
            raise ValueError('Found missing values. Drop rows with missing values (drop_incomplete) to continue.')

        x, normalizer = self.construct_inputs(df=df, normalizer=normalizer, target_type=target_type)
        y = construct_targets(df=df, target_type=target_type, one_hot=one_hot_targets)

        # Sampling inputs.
        if sampler is not None:
            x, y = sample_inputs(x=x, y=y, sampler=sampler, seed=seed)

        # Validate (input, target) pair sizes.
        if x.shape[0] != y.shape[0]:
            raise ValueError(f'Found inconsistent sizes between input and target pairs: {x.shape[0]} vs {y.shape[0]}')

        return x, y, normalizer
