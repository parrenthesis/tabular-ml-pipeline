from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def train_and_tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    numeric: List[str],
    categorical: List[str],
    model_type: str = "rf",
    cv_splits: int = 5,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train and tune a model (RandomForest or XGBoost) with cross-validation and return best estimator and results.
    Args:
        X: Feature DataFrame
        y: Target Series
        numeric: List of numeric feature names
        categorical: List of categorical feature names
        model_type: 'rf' or 'xgb'
        cv_splits: Number of CV splits
        random_state: Random seed
    Returns:
        best_estimator: Trained best estimator
        results: Dict of best params and CV/test scores
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                StandardScaler(),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
        ]
    )
    if model_type == "rf":
        param_grid = {
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_leaf": [1, 5, 10],
            "clf__n_estimators": [100, 200],
        }
        clf = RandomForestClassifier(random_state=random_state)
    elif model_type == "xgb":
        param_grid = {
            "clf__max_depth": [3, 5, 7],
            "clf__min_child_weight": [1, 3, 5],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
            "clf__reg_alpha": [0, 1],
            "clf__reg_lambda": [1, 10],
            "clf__n_estimators": [100, 200],
        }
        clf = XGBClassifier(eval_metric="logloss", random_state=random_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    pipe = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X, y)
    best_estimator = grid.best_estimator_
    y_pred = best_estimator.predict(X)
    y_proba = (
        best_estimator.predict_proba(X)[:, 1]
        if hasattr(best_estimator, "predict_proba")
        else None
    )
    report = classification_report(y, y_pred, output_dict=True)
    roc = roc_auc_score(y, y_proba) if y_proba is not None else None
    pr_auc = average_precision_score(y, y_proba) if y_proba is not None else None
    acc = accuracy_score(y, y_pred)
    results = {
        "best_params": grid.best_params_,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "report": report,
    }
    return best_estimator, results
