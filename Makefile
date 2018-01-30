.PHONY: test

test:
	python -m unittest discover

coverage:
	coverage run -m unittest discover ./test; coverage report -m

profile:
	python -m vmprof --no-native --lines -o prof.txt docs/benchmark.py 10000 1
	vmprofshow prof.txt --lines | less
