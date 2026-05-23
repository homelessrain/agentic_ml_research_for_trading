from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

from models.single_asset_model import SingleAssetModel
from utils.ml_data_processor import MinuteAndDailyMLDataProcessor
from utils.evaluator import BinaryClassificationEvaluator



class DirectionModel(SingleAssetModel):
    """
    This class is responsible for training a direction model.
    """
    def __init__(self, symbol: str):
        super().__init__(symbol)

        self._ml_data_processor = MinuteAndDailyMLDataProcessor(symbol)

        self._model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            enable_categorical=True,
        )

        self._evaluator = BinaryClassificationEvaluator()

    def train(self, start_datestr: str, end_datestr: str):

        # Fetch training data
        full_df = self._ml_data_processor.run(start_datestr, end_datestr, drop_label_na=True)
        feature_cols = self._ml_data_processor.feature_columns
        label_col = self._ml_data_processor.label_column
        df = full_df.loc[:, feature_cols + [label_col]]

        # Add internal data for debugging
        self._training_full_df = full_df
        self._training_feature_cols = feature_cols
        self._training_label_col = label_col

        # Split data into training and validation sets by random sampling
        train_df, val_df = cast(
            tuple[pd.DataFrame, pd.DataFrame],
            train_test_split(df, test_size=0.1, random_state=42),
        )

        X_train = train_df.loc[:, feature_cols]
        y_train = train_df.loc[:, label_col]
        X_val = val_df.loc[:, feature_cols]
        y_val = val_df.loc[:, label_col]

        # Train the model
        self._model.fit(X_train, y_train)

        # Evaluate the model on training data
        y_train_pred = self._model.predict(X_train)
        y_train_pred_prob = self._model.predict_proba(X_train)[:, 1]
        training_metrics = self._evaluator.evaluate(y_train, y_train_pred, y_train_pred_prob)

        # Evaluate the model on validation data, but still from the same time period, not walk-forward validation
        y_val_pred = self._model.predict(X_val)
        y_val_pred_prob = self._model.predict_proba(X_val)[:, 1]
        validation_metrics = self._evaluator.evaluate(y_val, y_val_pred, y_val_pred_prob)

        return training_metrics, validation_metrics

    def predict(self, start_datestr: str, end_datestr: str, drop_label_na: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Run prediction on a given time period.

        This function can be used for both batch prediction and online prediction.
        For online prediction, run predict(today datestr, today datestr) and get the last row for prediction on the latest bar.
        """

        # Fetch prediction data. Here we don't drop the rows with NA labels because we want to predict for most recent bars.
        full_df = self._ml_data_processor.run(start_datestr, end_datestr, drop_label_na=drop_label_na)
        feature_cols = self._ml_data_processor.feature_columns
        label_col = self._ml_data_processor.label_column

        # Add internal data for debugging
        self._prediction_full_df = full_df
        self._prediction_feature_cols = feature_cols
        self._prediction_label_col = label_col

        X_pred = full_df.loc[:, feature_cols]

        # Run prediction
        y_pred = self._model.predict(X_pred)
        prob_pred = self._model.predict_proba(X_pred)

        return y_pred, prob_pred[:, 1]

    def test(self, start_datestr: str, end_datestr: str) -> dict:
        """
        Test the model on a given time period.
        """

        y_test_pred, y_test_prob_pred = self.predict(start_datestr, end_datestr, drop_label_na=True)
        y_test = self._prediction_full_df.loc[:, self._prediction_label_col]
        testing_metrics = self._evaluator.evaluate(y_test, y_test_pred, y_test_prob_pred)

        return testing_metrics
