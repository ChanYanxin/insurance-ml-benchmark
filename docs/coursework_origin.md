# Coursework origin and refactoring notes

This repository was refactored from selected exercises in an *Introduction to Data Science* course project.

The original work covered:

- exploratory data analysis;
- decision trees;
- K-means clustering;
- linear regression;
- support vector machines;
- neural networks.

For a public engineering portfolio, the code was reorganized around reusable sklearn pipelines, explicit train/test separation, tests, a command-line experiment runner, and CI.

## Important modeling change: target leakage prevention

Some educational exercises intentionally expose labels or derived target columns in the same prepared table. In this refactor, the classification target is derived from medical charges, but `charges` is **strictly excluded from input features**. This prevents a model from learning the answer directly from the variable used to define the target.

## Original benchmark observations

On the supplied course split, the original notebooks reported:

- decision tree charge-group classification reaching about **94% accuracy** at depth 4;
- multivariate linear regression with a test **MAE of about 4.3k** monetary units;
- age-only linear regression performing substantially worse (about **8.9k MAE**), showing the value of multivariate features.

These numbers are documented as historical coursework results rather than guaranteed outputs of the refactored public demo, because the original dataset is not redistributed.
