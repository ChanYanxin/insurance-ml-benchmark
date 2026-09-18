import pandas as pd
import pytest
from insurance_ml.data import add_high_cost_target, load_insurance_csv


def test_load_and_target(tmp_path):
    p = tmp_path / "sample.csv"
    pd.DataFrame({
        "age": [20, 40],
        "sex": ["female", "male"],
        "bmi": [22.0, 31.0],
        "children": [0, 2],
        "smoker": ["no", "yes"],
        "region": ["northwest", "southeast"],
        "charges": [3000.0, 35000.0],
    }).to_csv(p, index=False)
    df = add_high_cost_target(load_insurance_csv(p))
    assert df["high_cost"].tolist() == [0, 1]


def test_missing_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"age": [20]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_insurance_csv(p)
