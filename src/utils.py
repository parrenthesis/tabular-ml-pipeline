import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Any

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Set up logging to file and console.
    Args:
        log_dir: Directory to save log files
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def plot_feature_importances(
    importances: np.ndarray,
    feature_names: List[str],
    model_name: str,
    out_dir: str = "plots",
    show: bool = False,
    save: bool = True
) -> str:
    """
    Plot and optionally save/show feature importances.
    Args:
        importances: Array of feature importances
        feature_names: List of feature names
        model_name: Name of the model (for filename)
        out_dir: Directory to save plot
        show: If True, display plot inline (for notebooks)
        save: If True, save plot to disk
    Returns:
        Path to saved plot (if saved), else None
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 3))
    plt.bar([feature_names[i] for i in sorted_idx[:10]], importances[sorted_idx[:10]])
    plt.title(f"Top 10 Feature Importances ({model_name})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"feature_importances_{model_name.replace(' ', '_')}.png")
    if save:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path)
    if show:
        plt.show()
    else:
        plt.close()
    return out_path if save else None


def plot_precision_recall_curve(
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray,
    model_name: str,
    out_dir: str = "plots",
    show: bool = False,
    save: bool = True
) -> str:
    """
    Plot and optionally save/show precision-recall vs threshold curve.
    Args:
        precisions: Array of precision values
        recalls: Array of recall values
        thresholds: Array of thresholds
        model_name: Name of the model (for filename)
        out_dir: Directory to save plot
        show: If True, display plot inline (for notebooks)
        save: If True, save plot to disk
    Returns:
        Path to saved plot (if saved), else None
    """
    import os
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, precisions[:-1], label='Precision')
    plt.plot(thresholds, recalls[:-1], label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Precision/Recall vs Threshold ({model_name})')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"pr_curve_{model_name.replace(' ', '_')}.png")
    if save:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_path)
    if show:
        plt.show()
    else:
        plt.close()
    return out_path if save else None 