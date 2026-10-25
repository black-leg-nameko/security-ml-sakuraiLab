import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def select_threshold(y, scores, max_fpr: float) -> float:
    y, scores = np.asarray(y), np.asarray(scores)
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must lie in [0, 1]")
    if set(np.unique(y)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Threshold selection requires both classes and finite scores")
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= max_fpr)
    # ROC thresholds descend; ties keep the more conservative operating point.
    best = eligible[np.argmax(tpr[eligible])]
    threshold = thresholds[best]
    return float(threshold if np.isfinite(threshold) else np.nextafter(scores.max(), np.inf))


def binary_metrics(y, scores, threshold: float = 0.5) -> dict:
    prediction = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "false_positive_rate": float(fp / (tn + fp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def feature_importance(model, feature_names=None) -> pd.Series:
    classifier = model.named_steps["classifier"]
    if "tfidf" in model.named_steps:
        feature_names = model.named_steps["tfidf"].get_feature_names_out()
    if hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_[0])
    else:
        values = classifier.feature_importances_
    return pd.Series(values, index=feature_names, name="importance").sort_values(ascending=False)


def save_report(output_dir, config, audits, studies, splits, metrics, hashes, revision):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": asdict(config),
        "positive_class": {"phishing": "UCI Result = -1", "spam": "enron1/spam"},
        "upstream_revision": revision,
        "dataset_sha256": hashes,
        "data_audit": {name: asdict(audit) for name, audit in audits.items()},
        "split_indices": {
            name: {k: v.tolist() for k, v in vars(s).items()} for name, s in splits.items()
        },
        "search": {
            name: {"best_cv_f1": study.best_value, "params": study.best_params}
            for name, study in studies.items()
        },
        "test_metrics": metrics,
        "versions": {
            name: version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "optuna")
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for name, study in studies.items():
        study.trials_dataframe().to_csv(output_dir / f"{name}_trials.csv", index=False)
    return report
