from abc import ABC, abstractmethod
import os

import numpy as np
import pandas as pd


class LabelTransformer(ABC):
    """
    This class is responsible for transforming bar data into ML-ready labels.
    """
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform bar data into ML-ready labels.

        Note that df here can be at arbitrary time granularity, e.g. 1 minute, 1 hour, 1 day, etc.
        """
        return df


class DirectionLabelTransformer(LabelTransformer):
    """
    This class is responsible for transforming bar data into ML-ready direction labels.
    """
    def __init__(
        self,
        price_column: str = 'close',
        label_column: str = 'label',
        positive_label: str = 'up',
        up_delta: float = 0.01,
        down_delta: float = 0.01,
        lookforward_period: int = 120,
    ):
        super().__init__()
        self._price_column = price_column
        self._label_column = label_column
        self._positive_label = positive_label
        assert positive_label in ['up', 'down'], "positive_label must be either 'up' or 'down'"
        self._up_delta = up_delta
        self._down_delta = down_delta
        self._lookforward_period = lookforward_period

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform bar data into ML-ready labels, in place.
        """

        df[f'max_up_{self._lookforward_period}_price'] = df[self._price_column].rolling(self._lookforward_period).max().shift(-self._lookforward_period)
        df[f'max_down_{self._lookforward_period}_price'] = df[self._price_column].rolling(self._lookforward_period).min().shift(-self._lookforward_period)

        df[f'max_up_{self._lookforward_period}_delta_pct'] = (df[f'max_up_{self._lookforward_period}_price'] - df[self._price_column]) / df[self._price_column]
        df[f'max_down_{self._lookforward_period}_delta_pct'] = (df[self._price_column] - df[f'max_down_{self._lookforward_period}_price']) / df[self._price_column]

        up_col   = f'max_up_{self._lookforward_period}_delta_pct'
        down_col = f'max_down_{self._lookforward_period}_delta_pct'
        valid    = df[f'max_up_{self._lookforward_period}_price'].notna()

        if self._positive_label == 'up':
            condition = (df[up_col] >= self._up_delta) & (df[down_col] < self._down_delta)
        elif self._positive_label == 'down':
            condition = (df[up_col] < self._up_delta) & (df[down_col] >= self._down_delta)
        else:
            raise ValueError(f"Invalid positive_label: {self._positive_label}")

        df[self._label_column] = np.where(valid, condition.astype(int), np.nan)
        return df


class VolatilityLabelTransformer(LabelTransformer):
    """
    This class is responsible for transforming bar data into ML-ready direction labels.
    """
    def __init__(
        self,
        price_column: str = 'close',
        label_column: str = 'label',
        change_delta: float = 0.01,
        lookforward_period: int = 120,
    ):
        super().__init__()
        self._price_column = price_column
        self._label_column = label_column
        self._change_delta = change_delta
        self._lookforward_period = lookforward_period

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform bar data into ML-ready labels, in place.
        """

        df[f'max_up_{self._lookforward_period}_price'] = df[self._price_column].rolling(self._lookforward_period).max().shift(-self._lookforward_period)
        df[f'max_down_{self._lookforward_period}_price'] = df[self._price_column].rolling(self._lookforward_period).min().shift(-self._lookforward_period)

        df[f'max_up_{self._lookforward_period}_delta_pct'] = (df[f'max_up_{self._lookforward_period}_price'] - df[self._price_column]) / df[self._price_column]
        df[f'max_down_{self._lookforward_period}_delta_pct'] = (df[self._price_column] - df[f'max_down_{self._lookforward_period}_price']) / df[self._price_column]

        up_col   = f'max_up_{self._lookforward_period}_delta_pct'
        down_col = f'max_down_{self._lookforward_period}_delta_pct'
        valid    = df[f'max_up_{self._lookforward_period}_price'].notna()

        condition = (df[up_col] >= self._change_delta) | (df[down_col] >= self._change_delta)
        df[self._label_column] = np.where(valid, condition.astype(int), np.nan)

        return df
