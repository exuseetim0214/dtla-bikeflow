PYTHON ?= python3

.PHONY: setup download audit prepare train backtest dashboard collect-status test lint all

setup:
	$(PYTHON) -m pip install -e '.[ml,app,dev]'

download:
	$(PYTHON) -m bikeflow download

audit:
	$(PYTHON) -m bikeflow audit

prepare:
	$(PYTHON) -m bikeflow prepare

train:
	$(PYTHON) -m bikeflow train

backtest:
	$(PYTHON) -m bikeflow backtest

dashboard:
	streamlit run src/bikeflow/dashboard.py

collect-status:
	$(PYTHON) -m bikeflow collect-status

test:
	pytest

lint:
	ruff check .

all: download audit prepare train

