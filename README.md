# extratrees.py - Extremely Randomized Trees

This is a Python implementation of the Extremely Randomized Trees (ExtraTrees)
technique as proposed in the article
[Geurts, Ernst & Wehenkel (2005)][geurts2005].

## Goal / progress

This does not aim to be anywhere near a production-ready library.
Mostly, it is a study in optimization and an experiment as to how to
get the most performance in python without using numpy or other third-party 
compiled libraries.

It is now **much** slower than the `sklearn` implementation, possible due to
three big differences:

1. `sklearn` uses [pre-sorting][presort] of attributes
2. Its really fast Cython optimisations
3. Numpy can do fast operations on rows as well as columns, whereas due to the
   nature of python lists, only one 'axis' is fast to operate on.

Presorting in without using numpy or Cyton does not seem to improve the result 
here much, since the required transposes are _really_ slow.

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

Time complexity in number of trees seems (almost) linear, in number of samples
it is somehwere between `O(nlogn)` and `O(n^2)`.

Ensemble train times may be a bit out of date, I don't check them very
regularly.

To run the benchmarks, get the [MNIST-CSV dataset][pjreddie], and put the files 
in `docs/mnist/` as `mnist_train.csv` and `mnist_test.csv`.
They are not included in the repository here as I do not own them.


### CPython 3.6.3

Single trees:

| N samples | Trees  |  Accuracy | Train [s] | Eval [s] |
|-----------|--------|-----------|-----------|----------|
|      1000 |      1 |       .59 |      0.13 |     0.09 |
|     10000 |      1 |       .72 |      1.49 |     0.09 |
|     60000 |      1 |       .83 |     11.22 |     0.11 |

Ensembles (without bagging/boosting):

| N samples | Trees  |  Accuracy | Train [s] | Eval [s] |
|-----------|--------|-----------|-----------|----------|
|      1000 |     10 |       .79 |      1.49 |     0.62 |
|     10000 |     10 |       .91 |     20.64 |     0.85 |
|     60000 |     10 |       .95 |    164.45 |     1.14 |

### Pypy 3.5.3

Single trees:

| N samples |  Trees |  Accuracy | Train [s] | Eval [s] |
|-----------|--------|-----------|-----------|----------|
|      1000 |      1 |       .53 |      0.20 |     0.05 |
|     10000 |      1 |       .75 |      0.77 |     0.12 |
|     60000 |      1 |       .82 |      4.43 |     0.08 |

Ensembles (without bagging/boosting):

| N samples | Trees  |  Accuracy | Train [s] | Eval [s] |
|-----------|--------|-----------|-----------|----------|
|      1000 |     10 |       .78 |      0.72 |     0.29 |
|     10000 |     10 |       .90 |      5.67 |     0.27 |
|     60000 |     10 |       .95 |     50.85 |     0.43 |


### Dataset reference

The MNIST dataset in CSV, from [pjreddie.com][pjreddie] was originally proposed
in [LeCun et al (1998)][lecun1998].

[lecun1998]: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
[geurts2005]: http://orbi.ulg.ac.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf
[pjreddie]: https://pjreddie.com/projects/mnist-in-csv/
[sklearn]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
[presort]: ./docs/presort.md
