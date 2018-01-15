# extratrees.py - Extremely Randomized Trees

This is a Python implementation of the Extremely Randomized Trees (ExtraTrees)
technique as proposed in the article
[Geurts, Ernst & Wehenkel (2005)][geurts2005].

## Goal

This does not aim to be anywhere near a production-ready library.
Mostly, it is an exercise in optimization and an experiment as to how to
get some performance in python without using numpy or other third-party compiled
libraries.

Some useful notes:

- To get the most common values or relative frequencies, loops are faster than
  `collections.Counter`
- `math.log(x)` is about 30% faster than `math.log(x,2)`
- Vector multiplication using lists (`sum([a*b for a,b in c])`) is faster when
  eliminating zeros (if they occur) first (`sum([a*b for a,b in c if b])`)

## Installation

Just copy the `.py` file somewhere and run it.
There are no external dependencies, though the example script requires
`matplotlib` to plot results.

## Usage

### Data format

Since there is no numpy, one cannot supply `np.array` matrices containing
attributes and outputs. Instead, plain lists of lists are used.

If your data is formatted as numpy `X,Y` arrays, use `tolist`:

```python
data = (X.tolist(), Y.tolist())
```

### Example

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

tree = ExtraTree()
tree.fit((X, Y))
tree.predict(X)
```

See [example.py](docs/example.py) for an applied example.

## Performance

This benchmark is performed an Intel i5 7200U with parameters `k=sqrt(n_classes)` 
and `n_min=10`.
The table below shows results the MNIST digits dataset.
The _N samples_ column shows how many samples were used for training, these are
the first `n` of the original training set.
All 10.000 available test samples are used for testing.

To run the benchmarks, get the [MNIST-CSV dataset][pjreddie], and put the files 
in `docs/mnist/` as `mnist_train.csv` and `mnist_test.csv`.
They are not included in the repository here as I do not own them.

_Note:_ Validity of these numbers subject to change with updates. Also,
`sklearn.ensemble.ExtraTreesClassifier` seems to be about 60x faster.

### CPython 3.6.3

| N samples | Trees  |  Accuracy | Train [s] | Eval [s] |
|-----------|--------|-----------|-----------|----------|
|      1000 |      1 |       .56 |      0.39 |     0.07 |
|      1000 |     10 |       .79 |      3.36 |     0.06 |
|      1000 |    100 |       .87 |     33.79 |     5.48 |
|     10000 |      1 |       .73 |      4.18 |     0.08 |
|     10000 |     10 |       .90 |     47.04 |     0.76 |
|   All 60k |      1 |       .82 |     31.39 |     0.11 |
|   All 60k |     10 |       .94 |    324.99 |     0.96 |

### Pypy 3.5.3

| N samples | Trees  |  Accuracy | Train [s] | Eval [s] |
|-----------|--------|-----------|-----------|----------|
|      1000 |      1 |       .56 |      0.39 |     0.08 |
|      1000 |     10 |       .77 |      2.35 |     0.26 |
|      1000 |    100 |       .88 |     20.57 |     1.48 |
|     10000 |      1 |       .74 |      3.16 |     0.09 |
|     10000 |     10 |       .90 |     28.83 |     0.30 |
|     10000 |    100 |       .94 |    282.34 |     2.34 |
|   All 60k |      1 |       .82 |     24.75 |     0.10 |
|   All 60k |     10 |       .94 |    226.11 |     0.42 |


### Dataset reference

The MNIST dataset in CSV, from [pjreddie.com][pjreddie] was originally proposed
in [LeCun et al (1998)][lecun1998].

[lecun1998]: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
[geurts2005]: http://orbi.ulg.ac.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf
[pjreddie]: https://pjreddie.com/projects/mnist-in-csv/
[sklearn]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
