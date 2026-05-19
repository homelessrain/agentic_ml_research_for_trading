from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pandas_ta  # # importing pandas_ta is necessary for the ta method to work, even if it is not explicitly called


class FeatureTransformer(ABC):
    """
    This class is responsible for transforming bar data into ML-ready features.

    Any feature transformers should NEVER use future data to derive features.
    """
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform bar data into ML-ready features.
        """
        return df


class OLHCVFeatureTransformer(FeatureTransformer):
    """
    This class derives features from OHLCV columns.
    """
    def __init__(self):
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform bar data into ML-ready features.

        Bar data input can have arbitrary time granularity, e.g. 1 minute, 1 hour, 1 day, etc.
        """

        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        assert all(col in df.columns for col in required_columns), f"Some of the required columns ({', '.join(required_columns)}) are missing."

        feature_columns = []

        # ========================================================
        # Raw bar features
        # ========================================================
        num_recent_bars = 10
        for i in range(num_recent_bars):
            for col in set(required_columns) - set(['datetime']):
                feature_col_name = f'{col}_lag_{i}'
                feature_columns.append(feature_col_name)
                df[feature_col_name] = df[col].shift(i)

        # ========================================================
        # Normalized bar features
        # ========================================================
        # bar-over-bar change percentage features
        for col in set(required_columns) - set(['datetime']):
            diff_col_name = f'{col}_diff'
            df[diff_col_name] = df[col].diff()
            df[diff_col_name + '_ratio'] = 1 + (df[diff_col_name] / df[f'{col}_lag_1'])
            df[diff_col_name + '_ratio_log'] = np.log(df[diff_col_name + '_ratio'])
            feature_columns.append(diff_col_name + '_ratio')
            feature_columns.append(diff_col_name + '_ratio_log')

        # Moving-average based normalized features
        moving_average_windows = [5, 10, 20, 50, 100, 200]
        for col in set(required_columns) - set(['datetime']):
            for maw in moving_average_windows:
                ma_col_name = f'{col}_ma_{maw}'
                df[ma_col_name] = df[col].rolling(window=maw).mean()
                df[f'{col}_by_' + ma_col_name + '_ratio'] = 1 + (df[col] / df[ma_col_name])
                df[f'{col}_by_' + ma_col_name + '_ratio_log'] = np.log(df[f'{col}_by_{ma_col_name}_ratio'])
                feature_columns.append(f'{col}_by_{ma_col_name}_ratio')
                feature_columns.append(f'{col}_by_{ma_col_name}_ratio_log')

        # ========================================================
        # Technical indicator features
        # ========================================================
        df['close_rsi'] = df.ta.rsi(close='close')
        feature_columns.append('close_rsi')

        macd = df.ta.macd(close='close')
        df['close_macd']        = macd.iloc[:, 0]  # MACD line
        df['close_macd_hist']   = macd.iloc[:, 1]  # histogram
        df['close_macd_signal'] = macd.iloc[:, 2]  # signal line
        feature_columns.extend(['close_macd', 'close_macd_hist', 'close_macd_signal'])

        # ========================================================
        # Time-based features
        # ========================================================
        df['hour_of_day'] = df['datetime'].dt.hour.astype('category')
        df['day_of_week'] = df['datetime'].dt.dayofweek.astype('category')

        return df
