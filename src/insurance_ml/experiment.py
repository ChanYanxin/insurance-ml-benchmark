from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from .clustering import cluster_numeric_profiles
from .config import FEATURES, RANDOM_STATE, TARGET_REGRESSION
from .data import add_high_cost_target, load_insurance_csv
from .evaluate import evaluate_classifier, evaluate_regressor
from .models import build_classifier, build_regressor


def run_benchmark(data_path: str | Path) -> dict:
    df = add_high_cost_target(load_insurance_csv(data_path))
    x = df[FEATURES]

    x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(
        x,
        df["high_cost"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["high_cost"],
    )
    cls_results = {}
    for name in ["logistic", "decision_tree", "svm", "mlp"]:
        model = build_classifier(name)
        model.fit(x_train_c, y_train_c)
        cls_results[name] = evaluate_classifier(model, x_test_c, y_test_c).to_dict()

    x_train_r, x_test_r, y_train_r, y_test_r = train_test_split(
        x,
        df[TARGET_REGRESSION],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    reg_results = {}
    for name in ["linear", "random_forest"]:
        model = build_regressor(name)
        model.fit(x_train_r, y_train_r)
        reg_results[name] = evaluate_regressor(model, x_test_r, y_test_r).to_dict()

    clusters = cluster_numeric_profiles(df)
    return {
        "n_rows": len(df),
        "classification": cls_results,
        "regression": reg_results,
        "clustering": {
            "silhouette": clusters.silhouette,
            "centroids": clusters.centroids.round(3).to_dict(orient="records"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the insurance ML benchmark.")
    parser.add_argument("--data", required=True, help="Path to CSV with raw insurance schema")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    results = run_benchmark(args.data)
    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
