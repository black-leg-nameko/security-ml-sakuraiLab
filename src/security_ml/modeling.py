from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 101
    trials: int = 8
    folds: int = 3
    max_false_positive_rate: float = 0.01

    def __post_init__(self):
        if self.trials < 1 or self.folds < 2:
            raise ValueError("trials >= 1 and folds >= 2 are required")
        if not 0 <= self.max_false_positive_rate <= 1:
            raise ValueError("max_false_positive_rate must lie in [0, 1]")


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def split_indices(y, seed: int) -> Split:
    indices = np.arange(len(y))
    train_validation, test = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=seed,
        stratify=np.asarray(y)[train_validation],
    )
    return Split(train, validation, test)


def make_model(kind: str, params: dict[str, Any], seed: int) -> Pipeline:
    if kind == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=params["C"],
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if kind == "tree":
        return Pipeline([("classifier", DecisionTreeClassifier(**params, random_state=seed))])
    if kind == "spam":
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        min_df=2,
                        max_features=20_000,
                        sublinear_tf=True,
                        dtype=np.float32,
                    ),
                ),
                (
                    "classifier",
                    LGBMClassifier(
                        **params,
                        objective="binary",
                        verbosity=-1,
                        n_jobs=2,
                        random_state=seed,
                        deterministic=True,
                        force_col_wise=True,
                        importance_type="gain",
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model: {kind}")


def suggest_params(trial: optuna.Trial, kind: str) -> dict[str, Any]:
    if kind == "logistic":
        return {"C": trial.suggest_float("C", 1e-3, 100, log=True)}
    if kind == "tree":
        return {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 16),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-6, 1e-2, log=True),
        }
    if kind == "spam":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 60, 160, step=20),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    raise ValueError(f"Unknown model: {kind}")


def tune_model(kind: str, X_train, y_train, config: ExperimentConfig):
    cv = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)

    def objective(trial):
        model = make_model(kind, suggest_params(trial, kind), config.seed)
        with threadpool_limits(limits=2):
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="f1",
                n_jobs=1,
                error_score="raise",
            )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
    )
    study.optimize(objective, n_trials=config.trials, n_jobs=1)
    model = make_model(kind, study.best_params, config.seed)
    with threadpool_limits(limits=2):
        model.fit(X_train, y_train)
    return model, study
