all: lint black isort

setup: install pre-commit

install:
	@echo "Installing dependencies..."
	poetry install

pre-commit: install
	@echo "Setting up pre-commit..."
	poetry run pre-commit install -t commit-msg -t pre-commit

test: test-black test-flake8 test-isort

test-black:
	@echo "Checking format with black..."
	poetry run black --check src *.py

test-flake8:
	@echo "Checking format with flake8..."
	poetry run flake8 src *.py --count --statistics

test-isort:
	@echo "Checking format with isort..."
	poetry run isort --check --settings-path pyproject.toml src *.py

format:
	@echo "Formatting with black and isort..."
	poetry run black src *.py
	poetry run isort --settings-path pyproject.toml src *.py
