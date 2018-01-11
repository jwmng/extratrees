.PHONY: test

test:
	python -m unittest discover

coverage:
	coverage run -m unittest discover ./test; coverage report -m
