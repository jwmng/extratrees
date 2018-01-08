# extratrees.py - Extremely Randomized Trees

This is a Python implementation of the Extremely Randomized Trees (ExtraTrees)
technique as proposed in the article
[Geurts, Ernst & Wehenkel (2005)][geurts2005].

The implementation aims to remain close to the pseudocode given in the article
and does not require any external dependencies.

There are three versions here:

- `extratrees.py`: No type annotations, supports `pypy`
- `extratrees_py2_types.py`: Has python2-style type hints, supports `pypy`
- `extratrees_py3_types.py`: Has python3 type annotations, passes `mypy --strict`,
  does _not_ support `pypy`


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
from src.extratrees import Dataset, ExtraTree

# Classification problem with three observations:
#    (1, 0) => Class 1
#    (1, 1) => Class 0
#    (2, 1) => Class 2

X = ((1, 0),
     (1, 1),
     (2, 1))

# Labels for each sample
Y = (1, 0, 2)

data = Dataset(X, Y)
tree = ExtraTree(n_min=1)
tree.fit(data)
tree.predict(data.attributes)
```

See [example.py](docs/example.py) for an applied example.

## Performance

This benchmark is performed an Intel i5 7200U with setting `k=n_classes` and
`n_min=10`.
The table below shows results the MNIST digits dataset.
The _n samples_ column shows how many samples were used for training, these are
the first `n` of the original training set.
For evaluation, all 10.000 test samples are used.

|Interpreter   | N samples |  Accuracy | Trees  | Train [s] | Eval [s] |
|--------------|-----------|-----------|--------|-----------|----------|
|Python3.6     |      1000 |     0.639 |      1 |         9 |    0.010 |
|Python3.6     |      1000 |     0.792 |     10 |        94 |    0.065 |
|Pypy3, py3.5  |      1000 |     0.638 |      1 |         2 |    0.051 |
|Pypy3, py3.5  |      1000 |     0.773 |     10 |        17 |    0.123 |
|Pypy3, py3.5  |      1000 |     0.823 |    100 |       178 |    0.329 |
|Pypy3, py3.5  |     10000 |     0.813 |      1 |        30 |    0.127 |
|Pypy3, py3.5  |     10000 |     0.918 |     10 |       298 |    0.476 |
|Pypy3, py3.5  |       All |     0.873 |      1 |       299 |    0.151 |

To run the benchmarks, get the [MNIST-CSV dataset][pjreddie], and put the files 
in `docs/mnist/` as `mnist_train.csv` and `mnist_test.csv`.
They are not included in the repository here as I do not own them.


### Dataset reference

Mnist: The MNIST dataset in CSV, from [pjreddie.com][pjreddie], originally by
[LeCun et al (1998)][lecun1998]

[lecun1998]: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
[geurts2005]: http://orbi.ulg.ac.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf
[pjreddie]: https://pjreddie.com/projects/mnist-in-csv/
[sklearn]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
