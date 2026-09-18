from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_classifier(model, x_test: pd.DataFrame, y_test: pd.Series) -> ClassificationMetrics:
    pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(x_test)
    else:
        score = pred
    return ClassificationMetrics(
        accuracy=accuracy_score(y_test, pred),
        precision=precision_score(y_test, pred, zero_division=0),
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, score),
    )


def evaluate_regressor(model, x_test: pd.DataFrame, y_test: pd.Series) -> RegressionMetrics:
    pred = model.predict(x_test)
    return RegressionMetrics(
        mae=mean_absolute_error(y_test, pred),
        rmse=mean_squared_error(y_test, pred) ** 0.5,
        r2=r2_score(y_test, pred),
    )
