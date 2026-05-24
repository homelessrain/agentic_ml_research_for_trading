from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pandas_ta  # importing pandas_ta is necessary for the ta method to work, even if it is not explicitly called


class FeatureTransformer(ABC):
    """
    This class is responsible for transforming bar data into ML-ready features.

    Any feature transformers should NEVER use future data to derive features.
    """
    def __init__(self):
        self._feature_columns: list[str] = []

    @property
    def feature_columns(self) -> list[str]:
        """Column names added by the most recent transform() call."""
        return self._feature_columns

    def _validate_columns(self, df: pd.DataFrame, required: list[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

    def _register(self, name: str) -> str:
        """Track a feature column name and return it."""
        self._feature_columns.append(name)
        return name

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform bar data into ML-ready features.
        """
        return df


class OLHCVFeatureTransformer(FeatureTransformer):
    """
    Derives features from OHLCV columns.
    """
    def __init__(self):
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform bar data into ML-ready features.

        Bar data input can have arbitrary time granularity, e.g. 1 minute, 1 hour, 1 day, etc.

        Return a dataframe with both raw columns and ML-ready features and labels. Downstream code should use the label_column and feature_columns properties to access the columns.
        """
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        self._validate_columns(df, required_columns)
        self._feature_columns = []

        value_columns = [c for c in required_columns if c != 'datetime']

        # ========================================================
        # Raw bar features
        # ========================================================
        # Commented out: ablation showed raw lags hurt generalization (−0.036 AUC)
        # and cause near-perfect train AUC (overfitting signal).
        # num_recent_bars = 10
        # for i in range(num_recent_bars):
        #     for col in value_columns:
        #         df[self._register(f'{col}_lag_{i}')] = df[col].shift(i)

        # ========================================================
        # Normalized bar features
        # ========================================================
        # Commented out: diff-ratios are nearly neutral (−0.008 AUC) and
        # largely redundant with MA ratios.
        # for col in value_columns:
        #     diff_col = f'{col}_diff'
        #     df[diff_col] = df[col].diff()
        #     df[self._register(f'{diff_col}_ratio')]     = 1 + (df[diff_col] / df[f'{col}_lag_1'])
        #     df[self._register(f'{diff_col}_ratio_log')] = np.log(df[f'{diff_col}_ratio'])

        moving_average_windows = [5, 10, 20, 50, 100, 200]
        for col in value_columns:
            for maw in moving_average_windows:
                ma_col = f'{col}_ma_{maw}'
                df[ma_col] = df[col].rolling(window=maw).mean()
                df[self._register(f'{col}_by_{ma_col}_ratio')]     = 1 + (df[col] / df[ma_col])
                df[self._register(f'{col}_by_{ma_col}_ratio_log')] = np.log(df[f'{col}_by_{ma_col}_ratio'])

        # ========================================================
        # Technical indicator features
        # ========================================================
        # Commented out: ablation showed RSI/MACD are redundant with MA ratios
        # (+0.004 AUC improvement when removed).
        # for length in [14, 20, 30, 50, 100, 200]:
        #     df[self._register(f'close_rsi_{length}')] = df.ta.rsi(close='close', length=length)
        #
        # macd = df.ta.macd(close='close', length=12, slow=26, signal=9)
        # df[self._register('close_macd')]        = macd.iloc[:, 0]
        # df[self._register('close_macd_hist')]   = macd.iloc[:, 1]
        # df[self._register('close_macd_signal')] = macd.iloc[:, 2]

        # ========================================================
        # Time-based features
        # ========================================================
        df[self._register('hour_of_day')] = df['datetime'].dt.hour.astype('category')
        df[self._register('day_of_week')] = df['datetime'].dt.dayofweek.astype('category')

        return df
