# Pre-sorting for tree generation

Pre-sorting may speed up the tree, essentially bringing training time complexity
down to the optimal `O(n*log n)`.
The question why is asked e.g. [here][SO-presort], and the answer is stated
below.
This document is mostly for reference as to how (if ever) it will be
implemented.

## Use cases

Some things essential to most tree training algorithms:

1. Find the minimum or maximum for some attribute among the current training
   subset, e.g. when identifying a cutoff candidate

2. Find all points for which `x_n >= t`, essentially evaluating any split on the
   current subset

3. Identifying uniformity among the labels of the current subset, to return
   early and avoid unneeded Information gain computation

## Preparation

Assume some dataset, with `x[i]` the `i`-th datapoint and `x[i][j]` its `j`-th
attribute:

    X = [
         [1, 6],
         [3, 5],
         [5, 4]
        ]

It has associated class labels:

    Y = [1, 1, 1]


## Pre-sorting

This involves a few steps, and may be pretty memory intensive.
Skip (1) if `X` was supplied as its transpose:

1. Transpose `X`, so that each row contains the attribute values:
    
        Xt = [
              [1, 3, 5],
              [6, 5, 4]
             ]

2. Sort the indices of the new matrix by their attribute value:

        Xtn = [
               [0, 1, 2],
               [2, 1, 0]
              ]

The matrix `Xtn[j]` now gives *the index of each datapoint, ordered by attribute 
`j`*.

3. Use the transpose of `Xtn` as the attribute matrix, save `Xt` so that it can
   be looked up *fast*
   values.


## Maxima and minima

The minimum of `X[:][j]` is datapoint `Xtn[0][j]`.
To get it, retrieve `X[Xtn[0][j]][j]`. e.g. for attribute `1`:
    
    Xtn[1] == [2, 1, 0]
    Xtn[0][0] == 0
    X[0][1] = 4

Likewise, the maximum is at point `Xtn[-1][j]`, thus is `X[Xtn[-1][j]][j]`.
If we can lookup a list in constant time, we can get minima and maxima in `O(n)`
(plus the time the sorting took).


## Uniformity

Since uniformity implies `min(x) != max(x)`, the fast maxima and minima above
can be used to do this in constant time as well.

## Thresholding

To find *each point `x` in `X` for which `x[j] <= t`*,  we can use a similar 
strategy:

    for idx in Xtn[j]:
        if X[idx] >= threshold:
            break

    return Xtn[:idx], Xtn[idx:]

This is guaranteed to complete, and runs in linear time.


[SO-presort]: https://stackoverflow.com/questions/37754380/does-presorting-slow-down-training-of-large-decision-trees

## Notes

- At 10k datapoints and 780 attributes, transposing (using `zip`) and sorting 
  using `sorted` combined takes around one second (around 55% of original
  training time)

