import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve


class BinaryClassificationEvaluator:
    """
    This class is responsible for evaluating the performance of a binary classification model.
    """
    def __init__(self):
        pass

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        prob_pred: pd.Series | np.ndarray,
    ) -> dict:

        ret = {}

        if prob_pred is not None:

            # Calculate label stats
            positive_label_rate = float(np.mean(y_true))

            # Calculate prediction stats
            mean_predicted_prob = float(np.mean(prob_pred))

            # Calculate ROC AUC score
            roc_auc = roc_auc_score(y_true, prob_pred)

            # Calculate Precision-Recall AUC score
            pr_auc = average_precision_score(y_true, prob_pred)

            # Calculate log loss
            log_loss_score = log_loss(y_true, prob_pred)

            # Calculate overall calibration
            bias = mean_predicted_prob - positive_label_rate

            # Calculate calibration curve
            prob_true_per_bin, prob_pred_per_bin = calibration_curve(y_true, prob_pred, strategy='quantile', n_bins=20)

            # Calculate correlation between predicted probability and true probability per bin
            correlation = np.corrcoef(prob_true_per_bin, prob_pred_per_bin)[0, 1]

            ret.update({
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'log_loss': log_loss_score,
                'positive_label_rate': positive_label_rate,
                'mean_predicted_prob': mean_predicted_prob,
                'bias': bias,
                'calibration_correlation': correlation,
                'calibration_curve': [prob_true_per_bin, prob_pred_per_bin],
            })

        if y_pred is not None:

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            ret.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            })

        return ret
