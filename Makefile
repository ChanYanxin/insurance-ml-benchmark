.PHONY: install test lint demo benchmark

install:
	python -m pip install -r requirements-dev.txt
	python -m pip install -e .

test:
	pytest -q

lint:
	ruff check src tests scripts

demo:
	python scripts/generate_demo_data.py --output data/demo_insurance.csv

benchmark: demo
	python -m insurance_ml.experiment --data data/demo_insurance.csv --output artifacts/demo_results.json
