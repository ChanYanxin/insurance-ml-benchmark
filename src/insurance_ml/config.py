from __future__ import annotations

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_REGRESSION = "charges"
TARGET_CLASSIFICATION = "high_cost"
HIGH_COST_THRESHOLD = 25_000.0
RANDOM_STATE = 42
