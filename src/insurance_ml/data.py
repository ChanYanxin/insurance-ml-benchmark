from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import FEATURES, HIGH_COST_THRESHOLD, TARGET_REGRESSION

REQUIRED_COLUMNS = set(FEATURES + [TARGET_REGRESSION])


def load_insurance_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate an insurance dataset.

    Expected schema: age, sex, bmi, children, smoker, region, charges.
    Rows with missing values in required columns are removed. This mirrors the
    small-data coursework setting while keeping preprocessing explicit.
    """
    df = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    clean = df.loc[:, FEATURES + [TARGET_REGRESSION]].copy()
    clean = clean.dropna().reset_index(drop=True)

    clean["age"] = pd.to_numeric(clean["age"], errors="raise")
    clean["bmi"] = pd.to_numeric(clean["bmi"], errors="raise")
    clean["children"] = pd.to_numeric(clean["children"], errors="raise")
    clean[TARGET_REGRESSION] = pd.to_numeric(clean[TARGET_REGRESSION], errors="raise")

    return clean


def add_high_cost_target(df: pd.DataFrame, threshold: float = HIGH_COST_THRESHOLD) -> pd.DataFrame:
    """Create a binary high-cost label without exposing charges as an input feature."""
    result = df.copy()
    result["high_cost"] = (result[TARGET_REGRESSION] > threshold).astype(int)
    return result
