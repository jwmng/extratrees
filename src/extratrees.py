"""
extratrees.py - Implementation of Extremely Randomised Trees

Geurts, Ernst & Wehenkel (2005)
http://orbi.ulg.ac.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf

This implementation aims to remain as close as possible to the implementation
as given in the paper (in pseudocode).
Most deviations from the pseudocode names are made in orderto adhere to python
naming and style conventions.
"""

import math
import random
from collections import namedtuple
from statistics import variance


TIMES = {}
TIME0 = 0
MAXENTROPY = 1e10

Node = namedtuple('Node', ['split', 'left', 'right'])
Split = namedtuple('Split', ['attribute', 'cutoff'])


def _variance(values):
    if not values:
        return -MAXENTROPY
    return variance(values) if len(values) > 1 else 0.0


def _entropy(values):
    """
    Shannon entropy

    This implementation uses log(x,e) instead of log(x,2) because it's around
    30% faster in the python implementation.
    """
    if not values:
        return -MAXENTROPY

    hist = _histogram(values)

    # This is >10% faster than a list comprehension in most cases
    entropy_sum = 0
    log = math.log
    for val in hist:
        if val:
            entropy_sum += log(val)*val

    return -entropy_sum


def _is_uniform(vals):
    """ Returns true if all elements in `vals` are the same """
    first = vals[0]
    for other in vals[1:]:
        if first != other:
            return False
    return True


def _gini(values):
    """ Gini impurity """
    if not values:
        return -MAXENTROPY
    hist = _histogram(values)

    imp_sum = 0.0
    for val in hist:
        imp_sum += val*(1-val)

    return imp_sum


def _histogram(values, n_classes=None):
    """
    Return the relative frequency of each int between 0 and n_classes
    Will guess `n_classes` if not specified
    """
    n_classes = n_classes or max(values)+1
    if not values:
        return [0.0]*n_classes

    n_samples = len(values)
    plusval = (1/n_samples)

    hist = [0]*n_classes
    for val in values:
        hist[val] += plusval

    return hist


def _mean(values):
    """ Return the mean of `values` """
    if not values:
        return 0
    return sum(values)/len(values)


def _majority(values):
    """ Return the most common value in `values` """
    counts = [0]*(max(values)+1)
    for val in values:
        counts[val] += 1

    return counts.index(max(counts))


def _argmax(values):
    """ Return the index of the largest value in `values` """
    return values.index(max(values))


def _evaluate_rec(node, attributes):
    """ Recursively traverse a node with `attributes` until a leaf is found"""
    # This is slightly faster than `isinstance`, since all should have a `split`
    # field, while histograms do not
    try:
        node.split
    except AttributeError:
        return node

    next_ = node.left if _evaluate_cond(node.split, attributes) else node.right
    return _evaluate_rec(next_, attributes)


def _pick_random_split(subset, attribute):
    """ Pick an (extremely random) split cutoff for `attribute` """
    attribute_values = [sample[attribute] for sample in subset[0]]

    max_a, min_a = max(attribute_values), min(attribute_values)
    cut_point = min_a + random.random()*(max_a-min_a)
    return Split(attribute, cut_point)


def _evaluate_cond(split, attributes):
    """ Evaluate the split condition on `attributes`

    Returns `True` if the split condition is true on the attributes
    """
    attribute, cutoff = split
    return attributes[attribute] > cutoff


def _evaluate_split_labels(subset, split):
    """ Same as `_evaluate_split`, but only returns labels """
    attributes, outputs = subset

    all_idxs = (range(len(attributes)))
    left = [_evaluate_cond(split, attribute) for attribute in attributes]
    left_indices = [idx for idx in all_idxs if left[idx]]
    right_indices = [idx for idx in all_idxs if not left[idx]]

    left_outputs = tuple(outputs[idx] for idx in left_indices)
    right_outputs = tuple(outputs[idx] for idx in right_indices)

    return left_outputs, right_outputs


def _evaluate_split(subset, split):
    """" Evaluate split

    Returns a tuple(left, right) with `left` the samples in `subset` for
    which the split condition is True
    """
    # Partition the data
    # In theory, this is faster than checking if `idx` is in `left_indices`,
    # since the latter requires enumerating `left_indices` every time.
    attributes, outputs = subset
    all_idxs = (range(len(attributes)))
    left = [_evaluate_cond(split, attribute) for attribute in attributes]

    left_indices = [idx for idx in all_idxs if left[idx]]
    right_indices = [idx for idx in all_idxs if not left[idx]]

    left_attributes = tuple(attributes[idx] for idx in left_indices)
    right_attributes = tuple(attributes[idx] for idx in right_indices)

    left_outputs = tuple(outputs[idx] for idx in left_indices)
    right_outputs = tuple(outputs[idx] for idx in right_indices)

    return ((left_attributes, left_outputs), (right_attributes, right_outputs))


class ExtraTree(object):
    """ ExtraTree object """
    def __init__(self, k_value=None, n_min=2, criterion="entropy"):
        super(ExtraTree, self).__init__()
        self.k_value = k_value
        self.n_min = n_min
        self.n_classes = 0
        self.root_node = Node(None, None, None)
        self._fitted = False
        self._is_classifier = False
        try:
            self.criterion = {"entropy": _entropy, "gini": _gini}[criterion]
        except KeyError:
            raise ValueError("Criterion '%s' unknown" % criterion)

    def fit(self, training_set):
        """ Fit a single tree """
        self._init_build(training_set)
        root_node = self._build_extra_tree_rec(training_set)
        if not isinstance(root_node, Node):
            raise ValueError("No samples")

        self.root_node = root_node
        self._fitted = True

    def predict_proba(self, samples):
        """ Soft predictions """
        assert self._fitted, "Tree has not been fitted, call fit() first"
        return [_evaluate_rec(self.root_node, sample) for sample in samples]

    def predict(self, samples):
        """ Predict, absoluting probabilities to the largest one """
        soft_pred = self.predict_proba(samples)
        return [_argmax(val) if isinstance(val, list) else val for val in
                soft_pred]

    def is_classifier(self):
        """ True if `tree` is fitted on a classification problem

        Returns `False` if it was not fitted, or was fitted on a regression
        problem
        """
        return self._is_classifier

    def _split_node(self, subset):
        """
        Args:
            subset (list): The local learning subset S.
                `subset` should be an iterable of samples, with each sample an
                iterable of attributes. All samples must have the same length.

        Returns:
            Split (idx, cutoff): A split, where `idx` is the attribute index
            and `cutoff` the cutoff value `a_c`
        """
        # This looks slow, but seems to be the fastest way using only python.
        # It is a lot faster than using `random.shuffle` for picking without
        # replacement, anyway
        n_attributes = len(subset[0][0])
        k_value = self.k_value
        randint = random.randint
        avail = list(range(n_attributes))

        best_score = -MAXENTROPY
        best_split = None

        # For each `k_value` iterations, take a random attribute from
        # `available` and use it to build a split
        for _ in range(k_value):
            attribute = avail.pop(randint(0, len(avail)-1))
            split = _pick_random_split(subset, attribute)
            score = self._score_split(subset, split)
            if score > best_score:
                best_score = score
                best_split = split

        return best_split

    def _stop_split(self, subset):
        """ Evaluate stopping condition on `subset` """
        attributes, outputs = subset
        if len(outputs) < self.n_min:
            return True

        if _is_uniform(outputs):
            return True

        # This assures that there are at least two non-equal attributes
        if _is_uniform(attributes):
            return True

        return False

    def _make_leaf(self, training_set):
        """ Create a leaf node from available data """
        if self._is_classifier:
            return _histogram(training_set[1], self.n_classes)

        return _mean(training_set[1])

    def _build_extra_tree_rec(self, training_set):
        # Return a leaf
        if self._stop_split(training_set):
            return self._make_leaf(training_set)

        # Split and recursively add children
        split = self._split_node(training_set)
        left_data, right_data = _evaluate_split(training_set, split)

        left_node = self._build_extra_tree_rec(left_data)
        right_node = self._build_extra_tree_rec(right_data)
        return Node(split, left_node, right_node)

    def _init_build(self, training_set):
        """ Initialise building before calling recursive `build` """
        # Some checks
        attributes, outputs = training_set
        dims = set(len(sample) for sample in attributes)
        assert len(dims) == 1, "Inconsistent attribute sizes"
        assert len(attributes) == len(outputs)

        # If there are only integer classes, assume a classification problem
        # Note that booleans pass this test so `True`/`False` are valid classes
        _is_classifier = all(isinstance(val, int) for val in outputs)

        # The default k_values are sqrt(n_attributes) for classification and
        # n_attributes for regression, as per the paper
        if self.k_value is None:
            n_attributes = len(attributes[0])
            if _is_classifier:
                self.k_value = int(math.sqrt(n_attributes))
            else:
                self.k_value = n_attributes

        n_classes = max(outputs) + 1

        # If the outputs are neither all float float or all int, we cannot
        # handle them
        if not _is_classifier:
            assert all(isinstance(val, float)
                       for val in outputs), "Unknown output types"
        else:
            assert (all(cls in outputs for cls in
                        range(n_classes))), "Classes are incomplete"

        self._is_classifier = _is_classifier
        self.n_classes = n_classes

    def _score_split(self, subset, split):
        """ Calculate information gain of a potential split node """
        left_out, right_out = _evaluate_split_labels(subset, split)

        n_left = len(left_out)
        n_right = len(right_out)
        n_total = n_left + n_right
        ratio_left = (n_left/n_total)

        # Classification: Use either gini or shannon as entropy metric
        criterion = self.criterion
        if self._is_classifier:
            ent_subset = criterion(subset[1])
            ent_left = (ratio_left)*criterion(left_out)
            ent_right = (1-ratio_left)*criterion(right_out)
            return ent_subset-ent_left-ent_right

        # Regression: use variance as entropy metric
        subset_var = _variance(subset[1])
        return float((subset_var
                      - (n_left/n_total)*_variance(left_out)
                      - (n_right/n_total)*_variance(right_out))
                     / subset_var)


class ExtraForest(object):
    """ Ensemble of ExtraTrees """

    def __init__(self, n_trees=10, k_value=None, n_min=1, criterion="entropy"):
        super(ExtraForest, self).__init__()
        self.n_trees = n_trees
        self.k_value = k_value
        self.n_min = n_min
        self.trees = []
        self._is_classifier = False
        self.criterion = criterion

    def fit(self, training_set):
        """ Fit each tree in the ensemble (multi-threaded) """
        self.trees = []
        for _ in range(self.n_trees):
            tree = ExtraTree(k_value=self.k_value, n_min=self.n_min,
                             criterion=self.criterion)
            tree.fit(training_set)
            self.trees.append(tree)

        self._is_classifier = self.trees[0].is_classifier()
        self.k_value = self.trees[0].k_value

    def predict(self, samples):
        """ Voted, hard-predict the class/value of the `samples` """
        votes = [tree.predict(samples) for tree in self.trees]

        votes_per_sample = list(zip(*votes))
        if not self._is_classifier:
            vote_fun = _mean
        else:
            vote_fun = _majority

        return [vote_fun(sample_votes) for sample_votes in votes_per_sample]

    def __repr__(self):
        fitted_str = "not fitted" if not self.trees else "fitted"
        return "<Forest (%d trees), %s>" % (self.n_trees, fitted_str)
