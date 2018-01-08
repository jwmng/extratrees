# extratrees.py - Implementation of Extremely Randomized Trees

This is a Python implementation of the Extremely Randomized Trees (ExtraTrees)
technique as proposed in the article
[Geurts, Ernst & Wehenkel (2005)][geurts2005].

The implementation aims to remain close to the pseudocode given in the article
and does not require any external dependencies.
To run tests you will need `mypy` and the example required `matplotlib`.
It is tested against  Python 3.6 and uses full static type hinting 
(`mypy --strict`).

[geurts2005]: http://orbi.ulg.ac.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf

## Data

Since there is no numpy, one cannot supply `np.array` matrices containing
attributes and outputs. Instead, a `namedtuple` is used to contain datasets:

```python
Dataset = namedtuple('Dataset', ['attributes', 'outputs'])
```

If your data is formatted as numpy `X,Y` arrays, use `tolist`:

```python
Dataset = namedtuple('Dataset', [X.tolist(), Y.tolist()])
```

- The `attributes` field is a list of _samples_, which in turn are lists of
attributes. Samples must have consistent shapes and consist exclusively of
integers or floats.

- The `outputs` must be a list of outputs, one for each sample in `attributes`.
Outputs may be either integers (`0...n`, where `n` is the number of classes) or
floats (for regression problems)

Example:

```python
# Classification problem with three observations:
#    (1, 0) => Class 1
#    (1, 1) => Class 0
#    (2, 1) => Class 2

X = ((1, 0),
     (1, 1),
     (2, 1))

# Labels for each sample
Y = (1, 0, 2)
```

See [example.py](docs/example.py) for a practical example.

## Todo

- Benchmark
- Check parameter values/constants with paper
- Optimize the random split function 
- Parallel/threaded tree training
- Performance test
