.PHONY: test

test:
	mypy --strict --ignore-missing-imports src
	python -m unittest discover

coverage:
	coverage run -m unittest discover ./test; coverage report -m
