"""
extratrees_untyped.py - Untyped version of extratrees.py

For pypy use
"""

import math
import random
from collections import namedtuple, Counter
from statistics import variance

MAXENTROPY = 1e10

Node = namedtuple('Node', ['split', 'left', 'right'])
Dataset = namedtuple('Dataset', ['attributes', 'outputs'])

Split = namedtuple('Split', ['attribute', 'cutoff'])


def _variance(values):
    if not values:
        return -MAXENTROPY
    return float(variance(values) if len(values) > 1 else 0)


def _entropy(values):
    """ Shannon entropy """
    if not values:
        return -MAXENTROPY

    hist = _histogram(values)
    log_hist = [math.log(val, 2) if val != 0 else 0 for val in hist]
    return -sum([h_val*lh_val for h_val, lh_val in zip(hist, log_hist)])


def _histogram(values,
               n_classes=None):
    """
    Return the relative frequency of each int between 0 and n_classes
    Will guess `n_classes` if not specified
    """
    n_classes = n_classes or max(values)+1

    cnt = Counter(values)
    return [cnt.get(v, 0)/len(values) for v in range(n_classes)]


def _mean(values):
    """ Return the mean of `values` """
    return sum(values)/max(len(values), 1)


def _majority(values):
    """ Return the most common value in `values` """
    return Counter(values).most_common(1)[0][0]


def _argmax(values):
    """ Return the index of the largest value in `values` """
    return values.index(max(values))


def _evaluate_rec(node, attributes):
    """ Recursively traverse a node with `attributes` until a leaf is found"""
    if not isinstance(node, Node):
        return node

    next_ = node.left if _evaluate_cond(node.split, attributes) else node.right
    return _evaluate_rec(next_, attributes)


def _pick_random_split(subset, attribute):
    """ Pick an (extremely random) split cutoff for `attribute` """
    attribute_values = [sample[attribute] for sample in
                        subset.attributes]

    max_a, min_a = max(attribute_values), min(attribute_values)
    cut_point = min_a + random.random()*(max_a-min_a)
    return Split(attribute, cut_point)


def _evaluate_cond(split, attributes):
    """ Evaluate the split condition on `attributes`

    Returns `True` if the split condition is true on the attributes
    """
    attribute, cutoff = split
    return bool(attributes[attribute] > cutoff)


def _evaluate_split(subset, split):
    """" Evaluate split

    Returns a tuple(left, right) with `left` the samples in `subset` for
    which the split condition is True
    """
    # Partition the data
    # In theory, this is faster than checking if `idx` is in `left_indices`,
    # since the latter requires enumerating `left_indices` every time.
    # Still, there is room for optimisation here
    left_indices = [idx for idx in range(len(subset.attributes))
                    if _evaluate_cond(split, subset.attributes[idx])]

    right_indices = [idx for idx in range(len(subset.attributes))
                     if not _evaluate_cond(split, subset.attributes[idx])]

    left_attributes = tuple(subset.attributes[idx] for idx in left_indices)
    right_attributes = tuple(subset.attributes[idx] for idx in right_indices)
    left_outputs = tuple(subset.outputs[idx] for idx in left_indices)
    right_outputs = tuple(subset.outputs[idx] for idx in right_indices)

    return (Dataset(left_attributes, left_outputs),
            Dataset(right_attributes, right_outputs))


class ExtraTree(object):
    """ ExtraTree object """
    def __init__(self, k_value=None, n_min=1):
        super(ExtraTree, self).__init__()
        self.k_value = k_value
        self.n_min = n_min
        self.n_classes = 0
        self.root_node = Node(None, None, None)
        self._fitted = False
        self._is_classifier = False

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
        # select K attributes; this is arguably the fastest way to obtain a
        # number of random indices without replacement
        n_attributes = list(range(len(subset.attributes[0])))
        random.shuffle(n_attributes)

        if self.k_value is not None:
            candidate_attributes = n_attributes[:self.k_value]
        else:
            candidate_attributes = n_attributes

        candidate_splits = [_pick_random_split(subset, attribute)
                            for attribute in candidate_attributes]

        candidate_scores = [self._score_split(subset, split)
                            for split in candidate_splits]

        return candidate_splits[candidate_scores.index(max(candidate_scores))]

    def _stop_split(self, subset):
        """ Evaluate stopping condition on `subset` """
        if len(subset.outputs) < self.n_min:
            return True
        if len(set(tuple(attr) for attr in subset.attributes)) <= 1:
            return True
        if len(set(subset.outputs)) == 1:
            return True
        return False

    def _make_leaf(self, training_set):
        """ Create a leaf node from available data """
        if self._is_classifier:
            return _histogram(training_set.outputs, self.n_classes)

        return _mean(training_set.outputs)

    def _build_extra_tree_rec(self, training_set):
        """ Train an ExtraTree, recursively """
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
        dims = set(len(sample) for sample in training_set.attributes)
        assert len(dims) == 1, "Inconsistent attribute sizes"
        assert len(training_set.attributes) == len(training_set.outputs)

        # If there are only integer classes, assume a classification problem
        # Note that booleans pass this test so `True`/`False` are valid classes
        _is_classifier = all(isinstance(val, int)
                             for val in training_set.outputs)

        n_classes = max(training_set.outputs) + 1

        # If the outputs are neither all float float or all int, we cannot
        # handle them
        if not _is_classifier:
            assert all(isinstance(val, float)
                       for val in training_set.outputs), "Unknown output types"
        else:
            assert (all(cls in training_set.outputs for cls in
                        range(n_classes))), "Classes are incomplete"

        self.n_classes = n_classes
        self._is_classifier = _is_classifier

    def _score_split(self, subset, split):
        """ Calculate information gain of a potential split node """
        left_data, right_data = _evaluate_split(subset, split)

        n_left = len(left_data.outputs)
        n_right = len(right_data.outputs)
        n_total = n_left + n_right

        # Classifier: note that this is the Shannon information gain. The
        # information metric in the paper may be slightly different
        if self._is_classifier:
            ent_subset = _entropy(subset.outputs)
            ent_left = (n_left/n_total)*_entropy(left_data.outputs)
            ent_right = (n_right/n_total)*_entropy(right_data.outputs)
            return ent_subset-ent_left-ent_right

        # Regression: use variance information gain
        subset_var = _variance(subset.outputs)
        return float((subset_var
                      - (n_left/n_total)*_variance(left_data.outputs)
                      - (n_right/n_total)*_variance(right_data.outputs))
                     / subset_var)


class ExtraForest(object):
    """ Ensemble of ExtraTrees """

    def __init__(self, n_trees=10, k_value=None, n_min=1):
        super(ExtraForest, self).__init__()
        self.n_trees = n_trees
        self.k_value = k_value
        self.n_min = n_min
        self.trees = []
        self._is_classifier = False

    def fit(self, training_set):
        """ Fit each tree in the ensemble """
        self.trees = []
        for _ in range(self.n_trees):
            tree = ExtraTree(k_value=self.k_value, n_min=self.n_min)
            tree.fit(training_set)
            self.trees.append(tree)

        self._is_classifier = self.trees[0].is_classifier()

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
