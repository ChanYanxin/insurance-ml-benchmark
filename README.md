# Insurance ML Benchmark — Leakage-Safe Tabular Learning Pipelines

A small but fully engineered machine-learning project that turns an academic notebook workflow into a **reproducible sklearn benchmark** for mixed tabular data.

The project covers three complementary tasks:

- **classification** — identify high-cost insurance cases;
- **regression** — predict continuous medical charges;
- **clustering** — discover numerical customer profiles.

It was refactored from coursework in *Introduction to Data Science* at RWTH Aachen University. The public repository focuses on engineering quality rather than reproducing assignment questions or distributing course data.

## Why this refactor matters

The original exercises explored decision trees, SVMs, neural networks, linear regression and K-means in notebooks. This version turns that work into a reusable ML codebase with:

- sklearn `Pipeline` + `ColumnTransformer` preprocessing;
- explicit protection against **target leakage**;
- train/test evaluation with task-appropriate metrics;
- multiple model families behind a consistent interface;
- a CLI experiment runner;
- synthetic demo-data generation for reproducibility;
- unit tests and GitHub Actions CI.

## Architecture

```text
raw CSV
  │
  ├── schema validation / cleaning
  │
  ├── numeric pipeline: impute → standardize
  │
  └── categorical pipeline: impute → one-hot encode
          │
          ├── classification: Logistic / Tree / SVM / MLP
          ├── regression: Linear / Random Forest
          └── clustering: StandardScaler → KMeans
```

## Classification task

A binary target is created as:

```text
high_cost = charges > 25,000
```

Crucially, `charges` is **not used as an input feature**. Models only receive:

```text
age, bmi, children, sex, smoker, region
```

This design prevents target leakage and makes the classification experiment representative of a real prediction task.

Metrics:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC

## Regression task

The regression benchmark predicts continuous `charges` and reports:

- MAE
- RMSE
- R²

The original coursework notebooks showed that multivariate regression substantially outperformed an age-only baseline. See [`docs/coursework_origin.md`](docs/coursework_origin.md) for historical results and refactoring notes.

## Clustering task

K-means is applied to standardized numerical profiles (`age`, `bmi`, `children`). The experiment returns:

- cluster assignments;
- centroids transformed back to the original feature scale;
- silhouette score.

## Reference results on the original coursework dataset

Using the refactored leakage-safe pipeline on the original dataset (1,338 complete rows, fixed 80/20 split, `random_state=42`) produced:

| Task | Model | Main result |
|---|---|---:|
| High-cost classification | Decision Tree | **F1 0.861 · ROC-AUC 0.949 · Accuracy 96.3%** |
| High-cost classification | SVM | F1 0.690 · ROC-AUC 0.903 |
| Charge regression | Linear Regression | MAE 4,234 · R² 0.773 |
| Charge regression | Random Forest | **MAE 2,475 · R² 0.864** |
| Profile clustering | K-Means (k=3) | Silhouette 0.312 |

These are reference measurements from the original course dataset; the repository intentionally does not redistribute that dataset. The synthetic demo verifies the full pipeline and CI behavior rather than reproducing those exact numbers.

## Project structure

```text
insurance-ml-benchmark/
├── src/insurance_ml/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── evaluate.py
│   ├── clustering.py
│   └── experiment.py
├── scripts/
│   └── generate_demo_data.py
├── tests/
├── data/
│   └── README.md
├── docs/
│   └── coursework_origin.md
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt
└── Makefile
```

## Quick start

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

Generate demo data and run the full benchmark:

```bash
make benchmark
```

Or run the commands directly:

```bash
python scripts/generate_demo_data.py --output data/demo_insurance.csv
python -m insurance_ml.experiment \
  --data data/demo_insurance.csv \
  --output artifacts/demo_results.json
```

Run tests:

```bash
pytest -q
```

## Engineering decisions

**1. No course data in the repository.** The original dataset is not redistributed. A synthetic generator makes the repository runnable end-to-end.

**2. Preprocessing is part of each model pipeline.** This avoids fitting scalers/encoders on the full dataset before the split.

**3. No leakage from the target-defining variable.** `charges` is used to define `high_cost` but never enters the classifier feature matrix.

**4. Notebooks are not the production interface.** The experiment is callable from Python and the command line, which makes tests and CI straightforward.

## Tech stack

Python · pandas · NumPy · scikit-learn · pytest · Ruff · GitHub Actions

## Next extensions

Useful follow-ups would be nested cross-validation, hyperparameter search, calibration analysis, feature importance/SHAP, experiment tracking and model packaging for inference.
