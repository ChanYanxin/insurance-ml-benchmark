from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE
from .features import build_preprocessor


def build_classifier(name: str) -> Pipeline:
    """Build one of the benchmark classification pipelines."""
    key = name.lower()
    estimators: dict[str, BaseEstimator] = {
        "logistic": LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=RANDOM_STATE),
        "svm": SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        "mlp": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2_000, early_stopping=True, random_state=RANDOM_STATE),
    }
    if key not in estimators:
        raise ValueError(f"Unknown classifier '{name}'. Choose from {sorted(estimators)}")
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimators[key])])


def build_regressor(name: str) -> Pipeline:
    """Build one of the benchmark regression pipelines."""
    key = name.lower()
    estimators: dict[str, BaseEstimator] = {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if key not in estimators:
        raise ValueError(f"Unknown regressor '{name}'. Choose from {sorted(estimators)}")
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimators[key])])
