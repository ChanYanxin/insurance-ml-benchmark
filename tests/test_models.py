from pathlib import Path
import subprocess
import sys

from insurance_ml.data import add_high_cost_target, load_insurance_csv
from insurance_ml.config import FEATURES
from insurance_ml.models import build_classifier, build_regressor


def _demo(tmp_path: Path):
    path = tmp_path / "demo.csv"
    subprocess.run(
        [sys.executable, "scripts/generate_demo_data.py", "--rows", "240", "--output", str(path)],
        check=True,
    )
    return add_high_cost_target(load_insurance_csv(path))


def test_classifier_pipeline_fits(tmp_path):
    df = _demo(tmp_path)
    model = build_classifier("decision_tree")
    model.fit(df[FEATURES], df["high_cost"])
    assert len(model.predict(df[FEATURES].head(5))) == 5


def test_regressor_pipeline_fits(tmp_path):
    df = _demo(tmp_path)
    model = build_regressor("linear")
    model.fit(df[FEATURES], df["charges"])
    assert len(model.predict(df[FEATURES].head(5))) == 5
