.PHONY: test

test:
	mypy --ignore-missing-imports src
	python -m unittest discover

coverage:
	coverage run -m unittest discover ./test; coverage report -m
