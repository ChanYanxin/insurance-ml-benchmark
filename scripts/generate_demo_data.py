from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 65, n)
    bmi = np.clip(rng.normal(30.5, 6.0, n), 16, 50)
    children = rng.integers(0, 6, n)
    sex = rng.choice(["female", "male"], n)
    smoker = rng.choice(["no", "yes"], n, p=[0.80, 0.20])
    region = rng.choice(["northeast", "northwest", "southeast", "southwest"], n)

    base = 1400 + 245 * age + 320 * bmi + 420 * children
    smoking_effect = np.where(smoker == "yes", 20_000 + 410 * bmi, 0)
    noise = rng.normal(0, 3500, n)
    charges = np.maximum(900, base + smoking_effect + noise)

    return pd.DataFrame({
        "age": age,
        "sex": sex,
        "bmi": bmi.round(3),
        "children": children,
        "smoker": smoker,
        "region": region,
        "charges": charges.round(2),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--output", default="data/demo_insurance.csv")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate(args.rows).to_csv(path, index=False)
    print(f"Wrote {args.rows} rows to {path}")


if __name__ == "__main__":
    main()
