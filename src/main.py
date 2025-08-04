import os
import argparse
import numpy as np
import pandas as pd
from src.featurize import extract_features_and_clean
from src.model import train_and_tune_model
from src.predict import predict_with_threshold, find_threshold_for_precision
from src.utils import setup_logging, plot_feature_importances, plot_precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, precision_score

def run_pipeline(db_path, model, output_dir, log_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir=log_dir)
    X, y, numeric, categorical = extract_features_and_clean(db_path)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Train class balance: {y_train.value_counts(normalize=True).to_dict()}")
    logger.info(f"Test class balance: {y_test.value_counts(normalize=True).to_dict()}")

    models_to_run = []
    if model == 'all':
        models_to_run = ["RandomForest", "XGBoost"]
    elif model == 'rf':
        models_to_run = ["RandomForest"]
    elif model == 'xgb':
        models_to_run = ["XGBoost"]

    model_objs = {}
    if "RandomForest" in models_to_run:
        rf_model, rf_results = train_and_tune_model(
            X_train, y_train, numeric, categorical, model_type="rf"
        )
        logger.info(f"RandomForest best params: {rf_results['best_params']}")
        model_objs["RandomForest"] = rf_model
    if "XGBoost" in models_to_run:
        xgb_model, xgb_results = train_and_tune_model(
            X_train, y_train, numeric, categorical, model_type="xgb"
        )
        logger.info(f"XGBoost best params: {xgb_results['best_params']}")
        model_objs["XGBoost"] = xgb_model

    # Evaluate on test set
    for name, model in model_objs.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        logger.info(f"\n{name} Test Classification Report:\n{report}")
        logger.info(f"{name} Test ROC AUC: {roc:.3f}, PR AUC: {pr_auc:.3f}")
        # Feature importances
        clf = model.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            ohe = model.named_steps['preprocessor'].named_transformers_['cat']
            cat_names = ohe.get_feature_names_out(categorical)
            feat_names = numeric + list(cat_names)
            importances = clf.feature_importances_
            plot_feature_importances(importances, feat_names, name, out_dir=output_dir, show=False, save=True)
        # Precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        plot_precision_recall_curve(precisions, recalls, thresholds, name, out_dir=output_dir, show=False, save=True)
        # Threshold tuning
        target_precision = 0.7
        thresh, prec, rec = find_threshold_for_precision(y_test.values, y_proba, target_precision)
        y_pred_auto = (y_proba >= thresh).astype(int)
        logger.info(f"{name} Auto-tuned threshold for precision>={target_precision}: threshold={thresh:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, Positives={np.sum(y_pred_auto == 1)}")
        # Default threshold
        y_pred_default = (y_proba >= 0.5).astype(int)
        def_prec = precision_score(y_test, y_pred_default)
        def_recall = np.sum((y_pred_default == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1)
        logger.info(f"{name} Default threshold=0.5: Precision={def_prec:.3f}, Recall={def_recall:.3f}, Positives={np.sum(y_pred_default == 1)}")

def main():
    parser = argparse.ArgumentParser(description="Tabular ML Pipeline: Predicting Treatment Need (Synthetic Healthcare Data)")
    parser.add_argument('--db-path', type=str, default=os.path.join("data", "data.db"), help='Path to SQLite database file')
    parser.add_argument('--model', type=str, default='all', choices=['rf', 'xgb', 'all'], help='Model to train (rf, xgb, or all)')
    parser.add_argument('--output-dir', type=str, default='plots', help='Directory to save plots/results')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory to save log files')
    args = parser.parse_args()
    run_pipeline(args.db_path, args.model, args.output_dir, args.log_dir)

if __name__ == "__main__":
    main() 