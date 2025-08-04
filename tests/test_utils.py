import logging
import matplotlib
import numpy as np
from src.utils import (
    plot_feature_importances,
    plot_precision_recall_curve,
    setup_logging,
)

matplotlib.use("Agg")  # Prevent plot display during tests


def test_setup_logging():
    logger = setup_logging()
    assert isinstance(logger, logging.Logger)


def test_plot_feature_importances():
    import matplotlib.pyplot as plt

    importances = np.array([0.2, 0.8])
    feat_names = ["a", "b"]
    # Default: save only
    path = plot_feature_importances(importances, feat_names, "TestModel")
    assert path is not None and path.endswith(".png")
    # Show only (should not save)
    path2 = plot_feature_importances(
        importances, feat_names, "TestModel", show=True, save=False
    )
    assert path2 is None
    # Show and save
    path3 = plot_feature_importances(
        importances, feat_names, "TestModel", show=True, save=True
    )
    assert path3 is not None and path3.endswith(".png")
    plt.close("all")


def test_plot_precision_recall_curve():
    import matplotlib.pyplot as plt

    precisions = np.array([1, 0.8, 0.5])
    recalls = np.array([0.1, 0.5, 1.0])
    thresholds = np.array([0.2, 0.5])
    # Default: save only
    path = plot_precision_recall_curve(precisions, recalls, thresholds, "TestModel")
    assert path is not None and path.endswith(".png")
    # Show only (should not save)
    path2 = plot_precision_recall_curve(
        precisions, recalls, thresholds, "TestModel", show=True, save=False
    )
    assert path2 is None
    # Show and save
    path3 = plot_precision_recall_curve(
        precisions, recalls, thresholds, "TestModel", show=True, save=True
    )
    assert path3 is not None and path3.endswith(".png")
    plt.close("all")
