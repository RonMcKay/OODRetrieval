all: lint black isort

lint:
	flake8 --config=.flake8 --exit-zero .

black:
	black --check .

isort:
	isort --settings-path pyproject.toml --check .
