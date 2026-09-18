# Data

The original coursework dataset is **not redistributed** in this repository.

To run the project without private/course files, generate a schema-compatible synthetic demo dataset:

```bash
python scripts/generate_demo_data.py --output data/demo_insurance.csv
```

For the original experiment, provide a CSV with these columns:

```text
age, sex, bmi, children, smoker, region, charges
```

The benchmark never uses `charges` as a classifier input. It is used only to build the binary evaluation label (`charges > 25000`) and as the regression target.
