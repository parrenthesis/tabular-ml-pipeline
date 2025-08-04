from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import precision_recall_curve


def predict(model: BaseEstimator, df: pd.DataFrame) -> Any:
    """
    Predict using the trained model and feature DataFrame.
    Args:
        model: Trained sklearn model
        df: Feature DataFrame
    Returns:
        Model predictions
    """
    return model.predict(df)


def predict_with_threshold(
    model: BaseEstimator, X: pd.DataFrame, threshold: float = 0.5
) -> np.ndarray:
    """
    Predict class labels using a custom threshold on predicted probabilities.
    Args:
        model: Trained sklearn model
        X: Feature DataFrame
        threshold: Probability threshold for class 1
    Returns:
        Array of predicted class labels
    """
    y_proba = model.predict_proba(X)[:, 1]
    return (y_proba >= threshold).astype(int)


def find_threshold_for_precision(
    y_true: np.ndarray, y_proba: np.ndarray, target_precision: float
) -> Tuple[float, float, float]:
    """
    Find the lowest threshold that achieves at least the target precision.
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for class 1
        target_precision: Desired minimum precision
    Returns:
        threshold, achieved_precision, recall_at_threshold
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    idx = np.argmax(precisions >= target_precision)
    if idx == len(thresholds):
        idx = -1
    return thresholds[idx], precisions[idx], recalls[idx]
