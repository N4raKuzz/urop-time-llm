from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix
)
import numpy as np
from typing import Dict, Optional

class ClassificationMetrics:
    '''This class handles binary classification metrics with error handling'''

    @staticmethod
    def _safe_metric(metric_func, y_true, y_pred, default_value=np.nan, **kwargs):
        try:
            return metric_func(y_true, y_pred, **kwargs)
        except Exception:
            return default_value

    @staticmethod
    def _check_binary_input(y_true: np.ndarray) -> bool:
        unique_classes = np.unique(y_true)
        return len(unique_classes) == 2

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return ClassificationMetrics._safe_metric(accuracy_score, y_true, y_pred)

    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> Optional[float]:
        return ClassificationMetrics._safe_metric(precision_score, y_true, y_pred, average=average, zero_division=np.nan)

    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> Optional[float]:
        return ClassificationMetrics._safe_metric(recall_score, y_true, y_pred, average=average, zero_division=np.nan)

    @staticmethod
    def f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> Optional[float]:
        return ClassificationMetrics._safe_metric(f1_score, y_true, y_pred, average=average, zero_division=np.nan)

    @staticmethod
    def auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
        if not ClassificationMetrics._check_binary_input(y_true):
            return np.nan
        return ClassificationMetrics._safe_metric(roc_auc_score, y_true, y_prob)

    @staticmethod
    def auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> Optional[float]:
        if not ClassificationMetrics._check_binary_input(y_true):
            return np.nan
        return ClassificationMetrics._safe_metric(average_precision_score, y_true, y_prob)

    @staticmethod
    def conf_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[str]:
        conf_matrix = ClassificationMetrics._safe_metric(confusion_matrix, y_true, y_pred)
        if isinstance(conf_matrix, np.ndarray):
            return str(conf_matrix.tolist())
        return None

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Optional[float]]:
        """Calculate all classification metrics with error handling."""
        y_pred = (y_prob > 0.5).astype(int)
        metrics = {
            'accuracy': ClassificationMetrics.accuracy(y_true, y_pred),
            'precision': ClassificationMetrics.precision(y_true, y_pred),
            'recall': ClassificationMetrics.recall(y_true, y_pred),
            'f1_score': ClassificationMetrics.f1(y_true, y_pred),
            'auc_roc': ClassificationMetrics.auc_roc(y_true, y_prob),
            'auc_pr': ClassificationMetrics.auc_pr(y_true, y_prob),
            'conf_matrix': ClassificationMetrics.conf_matrix(y_true, y_pred)
        }
        # Only include non-None and non-NaN values
        return {k: v for k, v in metrics.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
