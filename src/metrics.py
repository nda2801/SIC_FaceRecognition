import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def calculate_eer(labels, scores):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels,
        scores,
    )
    false_negative_rate = 1.0 - true_positive_rate
    index = np.nanargmin(
        np.abs(false_positive_rate - false_negative_rate)
    )
    return float(
        (false_positive_rate[index] + false_negative_rate[index]) / 2
    ), float(thresholds[index])


def calculate_roc_metrics(labels, distances):
    labels = np.asarray(labels)
    distances = np.asarray(distances)
    scores = -distances
    auc = roc_auc_score(labels, scores)
    eer, eer_threshold = calculate_eer(labels, scores)
    return {
        "roc_auc": float(auc),
        "eer": eer,
        "eer_threshold": eer_threshold,
    }
