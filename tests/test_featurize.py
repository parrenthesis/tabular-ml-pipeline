import os
from src.featurize import extract_features_and_clean

def test_extract_features_and_clean():
    db_path = os.path.join(os.path.dirname(__file__), "../data/data.db")
    if not os.path.exists(db_path):
        import pytest
        pytest.skip("data.db not present")
    X, y, numeric, categorical = extract_features_and_clean(db_path)
    assert not X.empty
    assert not y.empty
    assert isinstance(numeric, list)
    assert isinstance(categorical, list)
