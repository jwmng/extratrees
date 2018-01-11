"""
extratrees.py - Implementation of Extremely Randomised Trees

Geurts, Ernst & Wehenkel (2005)
http://orbi.ulg.ac.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf

This implementation aims to remain as close as possible to the implementation
as given in the paper (in pseudocode).
Most deviations from the pseudocode names are made in orderto adhere to python
naming and style conventions.
"""

import time
import math
import random
from collections import namedtuple
from statistics import variance

TIMES = {}
TIME0 = 0
MAXENTROPY = 1e10

Node = namedtuple('Node', ['split', 'left', 'right'])
Split = namedtuple('Split', ['attribute', 'cutoff'])

# #### Temporary timing functions #####
def tic():
    global TIME0
    TIME0 = time.time()

def toc(name):
    global TIME0
    TIMES[name] = TIMES.get(name, 0) + (time.time() - TIME0)
    print(TIMES)
    tic()
######################################


def _variance(values):
    if not values:
        return -MAXENTROPY
    return (variance(values) if len(values) > 1 else 0.0)


def _entropy(values):
    """
    Shannon entropy

    This implementation uses log(x,e) instead of log(x,2) because it's around
    30% faster in the python implementation.
    """
    if not values:
        return -MAXENTROPY

    hist = _histogram(values)

    # This seems actually faster than list comprehensions
    entropy_sum = 0
    for val in hist:
        if val:
            entropy_sum += math.log(val)*val
    return -entropy_sum


def _histogram(values, n_classes=None):
    """
    Return the relative frequency of each int between 0 and n_classes
    Will guess `n_classes` if not specified
    """
    n_classes = n_classes or max(values)+1
    if not values:
        return [0.0]*n_classes

    n_samples = len(values)

    hist = [0]*n_classes
    for val in values:
        hist[val] += 1
    hist = [x/n_samples for x in hist]

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
    if not isinstance(node, Node):
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

    tic()
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
    def __init__(self, k_value=None, n_min=2):
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
        # select K attributes; this is (arguably) the fastest way to obtain a
        # number of random indices without replacement
        n_attributes = len(subset[0][0])
        attributes = list(range(n_attributes))
        random.shuffle(attributes)

        candidate_attributes = attributes[:self.k_value]

        candidate_splits = [_pick_random_split(subset, attribute)
                            for attribute in candidate_attributes]

        candidate_scores = [self._score_split(subset, split)
                            for split in candidate_splits]

        best_candidate_idx = _argmax(candidate_scores)
        return candidate_splits[best_candidate_idx]

    def _stop_split(self, subset):
        """ Evaluate stopping condition on `subset` """
        attributes, outputs = subset
        if len(outputs) < self.n_min:
            return True
        if len(set(tuple(attr) for attr in attributes)) <= 1:
            return True
        if len(set(outputs)) == 1:
            return True
        return False

    def _make_leaf(self, training_set):
        """ Create a leaf node from available data """
        if self._is_classifier:
            return _histogram(training_set[1], self.n_classes)

        return _mean(training_set[1])

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

        # Classifier: note that this is the Shannon information gain. The
        # information metric in the paper may be slightly different
        if self._is_classifier:
            ent_subset = _entropy(subset[1])
            ent_left = (ratio_left)*_entropy(left_out)
            ent_right = (1-ratio_left)*_entropy(right_out)
            return ent_subset-ent_left-ent_right

        # Regression: use variance information gain
        subset_var = _variance(subset[1])
        return float((subset_var
                      - (n_left/n_total)*_variance(left_out)
                      - (n_right/n_total)*_variance(right_out))
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
