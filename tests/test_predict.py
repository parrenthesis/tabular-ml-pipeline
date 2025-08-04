import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import pytest
from src.predict import find_threshold_for_precision, predict_with_threshold, predict

def test_find_threshold_for_precision():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.4, 0.8, 0.7, 0.6, 0.2, 0.9])
    threshold, precision, recall = find_threshold_for_precision(y_true, y_proba, target_precision=0.7)
    assert 0 <= threshold <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1

def test_find_threshold_for_precision_no_match():
    # No threshold can achieve precision >= 1.1
    y_true = np.array([0, 1, 1, 0])
    y_proba = np.array([0.2, 0.3, 0.4, 0.5])
    threshold, precision, recall = find_threshold_for_precision(y_true, y_proba, target_precision=1.1)
    assert isinstance(threshold, float)
    assert isinstance(precision, float)
    assert isinstance(recall, float)

def test_find_threshold_for_precision_idx_edge():
    # All precisions below target, so idx == len(thresholds)
    y_true = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])
    # Target precision is higher than any achievable
    threshold, precision, recall = find_threshold_for_precision(y_true, y_proba, target_precision=1.0)
    assert isinstance(threshold, float)
    assert isinstance(precision, float)
    assert isinstance(recall, float)

def test_predict_with_threshold():
    # Fit a simple model and test thresholding
    X = pd.DataFrame({'a': [0, 1, 0, 1], 'b': [1, 0, 1, 0]})
    y = np.array([0, 1, 0, 1])
    clf = RandomForestClassifier().fit(X, y)
    preds_05 = predict_with_threshold(clf, X, threshold=0.5)
    preds_09 = predict_with_threshold(clf, X, threshold=0.9)
    assert preds_05.shape == y.shape
    assert preds_09.shape == y.shape
    assert set(preds_05).issubset({0, 1})
    assert set(preds_09).issubset({0, 1})

def test_predict():
    X = pd.DataFrame({'a': [0, 1, 0, 1], 'b': [1, 0, 1, 0]})
    y = np.array([0, 1, 0, 1])
    clf = LogisticRegression().fit(X, y)
    preds = predict(clf, X)
    assert preds.shape == y.shape
    assert set(preds).issubset({0, 1})

def test_predict_with_threshold_no_predict_proba():
    class DummyNoProba(BaseEstimator):
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X))
    X = pd.DataFrame({'a': [0, 1], 'b': [1, 0]})
    model = DummyNoProba().fit(X, np.array([0, 1]))
    with pytest.raises(AttributeError):
        predict_with_threshold(model, X, threshold=0.5) 