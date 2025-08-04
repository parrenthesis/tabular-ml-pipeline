import numpy as np
import pandas as pd
import pytest
from src.model import train_and_tune_model

def make_toy_data():
    X = pd.DataFrame({
        'num1': np.random.randn(20),
        'num2': np.random.randn(20),
        'cat1': np.random.choice(['a', 'b'], 20),
        'cat2': np.random.choice(['x', 'y'], 20),
    })
    y = np.random.choice([0, 1], 20)
    numeric = ['num1', 'num2']
    categorical = ['cat1', 'cat2']
    return X, y, numeric, categorical

def test_train_and_tune_model_rf():
    X, y, numeric, categorical = make_toy_data()
    model, results = train_and_tune_model(X, y, numeric, categorical, model_type='rf', cv_splits=2)
    assert hasattr(model, 'predict')
    assert 'best_params' in results
    assert 'roc_auc' in results

def test_train_and_tune_model_xgb():
    X, y, numeric, categorical = make_toy_data()
    model, results = train_and_tune_model(X, y, numeric, categorical, model_type='xgb', cv_splits=2)
    assert hasattr(model, 'predict')
    assert 'best_params' in results
    assert 'roc_auc' in results

def test_train_and_tune_model_invalid_type():
    X, y, numeric, categorical = make_toy_data()
    with pytest.raises(ValueError):
        train_and_tune_model(X, y, numeric, categorical, model_type='invalid', cv_splits=2) 