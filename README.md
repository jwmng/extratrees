# extratrees.py - Implementation of Extremely Randomized Trees

This is a Python implementation of the Extremely Randomized Trees (ExtraTrees)
technique as proposed in the article
[Geurts, Ernst & Wehenkel (2005)][geurts2005].

The implementation aims to remain close to the pseudocode given in the article
and does not require any external dependencies.
To run tests you will need `mypy` and the example required `matplotlib`.
It is tested against  Python 3.6 and uses full static type hinting 
(`mypy --strict`).
The benchmark script requires `sklearn` for the 'iris' dataset.

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

## Performance

This benchmark is performed an Intel i5 7200U with setting `k=n_classes` and
`n_min=10`.
The table below shows results the MNIST digits dataset.
The _Training_ column shows the number of training samples used.
Evaluation was done on all 10k samples in the Test dataset.

Interpreter          Training     Accuracy   Trees   Training [s]  Evaluation [s]
==================   ========    ========    =====   ===========   ==============
Python3.6                  1k       0.639        1         9.228            0.010
Python3.6                  1k       0.792       10        94.374            0.065
Pypy3, py3.5               1k       0.638        1         2.118            0.051
Pypy3, py3.5               1k       0.773       10        17.547            0.123
Pypy3, py3.5               1k       0.823      100       178.517            0.329
Pypy3, py3.5               10k      0.813        1        30.915            0.127
Pypy3, py3.5               10k      0.918       10       298.334            0.476
Pypy3, py3.5               All      0.873        1       299.831            0.151

To run the benchmarks, get the [MNIST-CSV dataset][pjreddie], and put the files 
in `docs/mnist/` as `mnist_train.csv` and `mnist_test.csv`.
They are not included in the repository here as I don not own them.


### Dataset reference

- Spiral: `spiral`: 2-class spiral data, see [benchmark.py](./docs/benchmark.py)
- Iris: `iris` from [`sklearn.datasets`](sklearn)
- Digits: `digits` from [`sklearn.datasets`](sklearn)
- Mnist: The MNIST dataset in CSV, from [pjreddie.com][pjreddie]

[pjreddie]: (https://pjreddie.com/projects/mnist-in-csv/)
[sklearn]: (http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets)
