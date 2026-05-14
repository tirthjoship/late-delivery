.PHONY: test test-cov lint typecheck setup check app

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=domain --cov=adapters --cov=application --cov-fail-under=90 --tb=short

lint:
	pre-commit run --all-files

typecheck:
	mypy domain/ adapters/ application/ --strict

setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	conda run -n supply-chain-ml pip install -e ".[dev]"
	pre-commit install

check: lint typecheck test-cov

app:
	streamlit run app/streamlit_app.py
