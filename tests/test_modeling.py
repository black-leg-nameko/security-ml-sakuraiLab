import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from security_ml.evaluation import binary_metrics, select_threshold
from security_ml.modeling import ExperimentConfig, make_model, split_indices


def test_splits_are_disjoint_complete_and_repeatable():
    y = np.tile([0, 1], 100)
    split = split_indices(y, 101)
    repeated = split_indices(y, 101)
    assert [len(split.train), len(split.validation), len(split.test)] == [120, 40, 40]
    assert sorted(np.concatenate(list(vars(split).values()))) == list(range(200))
    for name, indices in vars(split).items():
        np.testing.assert_array_equal(indices, getattr(repeated, name))
        assert y[indices].mean() == 0.5


def test_tfidf_does_not_learn_holdout_vocabulary_and_remains_sparse():
    model = make_model("spam", {"n_estimators": 2, "min_child_samples": 1}, seed=101)
    train = pd.Series(["meeting schedule", "meeting agenda", "prize winner", "prize free"])
    model.fit(train, [0, 0, 1, 1])
    vectorizer = model.named_steps["tfidf"]
    holdout = vectorizer.transform(["unseentoken prize"])
    assert "unseentoken" not in vectorizer.vocabulary_
    assert issparse(holdout)
    np.testing.assert_allclose(model.predict_proba(["unseentoken prize"]).sum(axis=1), 1)


def test_threshold_satisfies_validation_constraint_and_maximizes_recall():
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.1, 0.4, 0.8, 0.3, 0.6, 0.9])
    threshold = select_threshold(y, scores, max_fpr=0.25)
    metrics = binary_metrics(y, scores, threshold)
    assert threshold == 0.6
    assert metrics["false_positive_rate"] == 0.25
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert [metrics[k] for k in ("tn", "fp", "fn", "tp")] == [3, 1, 1, 2]


def test_tied_scores_can_abstain_from_all_positive_predictions():
    y = [0, 1, 0, 1]
    scores = [0.7] * 4
    threshold = select_threshold(y, scores, max_fpr=0)
    assert np.isfinite(threshold) and threshold > 0.7
    metrics = binary_metrics(y, scores, threshold)
    assert metrics["fp"] == metrics["tp"] == 0


@pytest.mark.parametrize("kwargs", [{"trials": 0}, {"folds": 1}, {"max_false_positive_rate": 1.1}])
def test_invalid_experiment_budget_or_operating_point_fails(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)
