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


def _variance(values):
    if not values:
        return -MAXENTROPY
    return variance(values) if len(values) > 1 else 0.0


def _entropy(hist):
    """
    Shannon entropy

    This implementation uses log(x,e) instead of log(x,2) because it's around
    30% faster in the python implementation.
    """
    entropy_sum = 0
    log = math.log

    h_sum = sum(hist)
    if not h_sum:
        return MAXENTROPY

    for val in hist:
        freq = val/h_sum
        try:
            entropy_sum += log(freq)*freq
        except ValueError:
            continue

    return -entropy_sum


def _is_uniform(vals):
    """ Returns true if all elements in `vals` are the same """
    first = vals[0]
    for other in vals[1:]:
        if first != other:
            return False
    return True


def _gini(hist):
    """ Gini impurity from histogram """
    imp_sum = 0.0
    h_sum = sum(hist)
    if not h_sum:
        return MAXENTROPY

    for val in hist:
        freq = val/h_sum
        imp_sum += freq*(1-freq)

    return imp_sum


def _histogram(values, n_classes):
    """
    Return the relative frequency of each int between 0 and n_classes
    Will guess `n_classes` if not specified
    """
    hist = [0]*n_classes
    if not values:
        return hist

    for val in values:
        hist[val] += 1

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

    # This is slightly faster than calling `isinstance` and more EAFP
    try:
        node.split
    except AttributeError:
        return node

    next_ = node.left if _evaluate_cond(node.split, attributes) else node.right
    return _evaluate_rec(next_, attributes)


def _pick_random_split(vals, attribute):
    """ Pick an (extremely random) split cutoff for `attribute` """
    max_a, min_a = max(vals), min(vals)
    if max_a == min_a:
        return None

    cut_point = min_a + random.random()*(max_a-min_a)
    return (attribute, cut_point)


def _evaluate_cond(split, attributes):
    """ Evaluate the split condition on `attributes`

    Returns `True` if the split condition is true on the attributes
    """
    return attributes[split[0]] > split[1]


def _evaluate_oneside(subset, split):
    """ Same as `_evaluate_split`, returns only labels and the point ratio """
    attributes, outputs = subset
    attr_idx, attr_thresh = split
    left_outputs = (output for idx, output in enumerate(outputs) if
                    attributes[idx][attr_idx] > attr_thresh)

    return left_outputs


def _evaluate_split(subset, split):
    """" Evaluate split

    Returns a tuple(left, right) with `left` the samples in `subset` for
    which the split condition is True
    """
    # Partition the data
    # In theory, this is faster than checking if `idx` is in `left_indices`,
    # since the latter requires enumerating `left_indices` every time.
    attributes, outputs = subset
    attr_idx, attr_thresh = split

    left = [attributes[idx][attr_idx] > attr_thresh
            for idx, _ in enumerate(outputs)]

    left_indices = [idx for idx, val in enumerate(left) if val]
    right_indices = [idx for idx, val in enumerate(left) if not val]

    left_attributes = tuple(attributes[idx] for idx in left_indices)
    right_attributes = tuple(attributes[idx] for idx in right_indices)

    left_outputs = tuple(outputs[idx] for idx in left_indices)
    right_outputs = tuple(outputs[idx] for idx in right_indices)

    return ((left_attributes, left_outputs), (right_attributes, right_outputs))


class ExtraTreeClassifier(object):
    """ ExtraTree object """
    def __init__(self, k_value=None, n_min=2, criterion="entropy"):
        super(ExtraTreeClassifier, self).__init__()
        self.k_value = k_value
        self.n_min = n_min
        self.n_classes = 0
        self.root_node = Node(None, None, None)
        self._fitted = False
        try:
            self.criterion = {"entropy": _entropy, "gini": _gini}[criterion]
        except KeyError:
            raise ValueError("Criterion '%s' unknown" % criterion)

    def fit(self, training_set):
        """ Fit a single tree """
        self._init_build(training_set)
        root_node = self._build_extra_tree_rec(training_set)

        if not isinstance(root_node, Node):
            raise ValueError("Training failed: Not a single split found")

        self.root_node = root_node
        self._fitted = True

    def predict_proba(self, samples):
        """ Soft predictions """
        assert self._fitted, "Tree has not been fitted, call fit() first"
        return [_evaluate_rec(self.root_node, sample) for sample in samples]

    def predict(self, samples):
        """ Predict, absoluting probabilities to the largest one """
        soft_pred = self.predict_proba(samples)
        return [_argmax(hist_) for hist_ in soft_pred]

    def _split_node(self, subset, subset_hist, subset_ent):
        """
        Args:
            subset (list): The local learning subset S.
                `subset` should be an iterable of samples, with each sample an
                iterable of attributes. All samples must have the same length.

        Returns:
            tuple (idx, cutoff): A split, where `idx` is the attribute index
            and `cutoff` the cutoff value `a_c`
        """
        # This looks slow, but seems to be the fastest way using only python.
        # It is a lot faster than using `random.shuffle` for picking without
        # replacement, anyway
        n_attributes = len(subset[0][0])
        k_value = self.k_value
        avail = list(range(n_attributes))

        best_score = -MAXENTROPY
        best_split = None
        best_ents = (None, None)

        # For each `k_value` iterations, take a (new) random attribute from
        # `available` and use it to build a split
        # See if this is faster
        for _ in range(k_value):
            attribute = avail.pop(int(random.random()*(len(avail)-1)))
            vals = [sample[attribute] for sample in subset[0]]

            if _is_uniform(vals):
                continue

            split = _pick_random_split(vals, attribute)

            score, *ents = self._score_split(subset, split, subset_hist,
                                             subset_ent)
            if score > best_score:
                best_score = score
                best_split = split
                best_ents = ents

        # No informative split found, just return the last one
        if best_split is None:
            return None, None, None

        return best_split, best_ents[0], best_ents[1]

    def _stop_split(self, subset):
        """
        Return true, if, for `subset`:
            1. There are fewer than `n_min` samples left
            2. All labels in `subset` are equal
            3. All attributes `subset` are equal

        The order is chosen empirically. Also, the uniformity check on
        attributes tends to be slower than on the others, especially with
        highly dimensional data.
        """
        attributes, outputs = subset
        if len(outputs) < self.n_min:
            return True

        if _is_uniform(outputs):
            return True

        if _is_uniform(attributes):
            return True

        return False

    def _make_leaf(self, training_set):
        """ Create a leaf node (histogram) from available data """
        return _histogram(training_set[1], self.n_classes)

    def _build_extra_tree_rec(self, training_set, training_set_hist=None,
                              training_set_ent=None):
        """ Recursively build the tree """

        # Create a label histogram if it is not known
        if training_set_hist is None:
            training_set_hist = _histogram(training_set[1], self.n_classes)

        # Same for entropy
        if training_set_ent is None:
            training_set_ent = self.criterion(training_set_hist)

        # The leaf for a classifier is the (pre-calculated) histogram
        if self._stop_split(training_set):
            return training_set_hist

        # Get the optimal split and partition data for child nodes
        split, left_ent, right_ent = self._split_node(training_set,
                                                      training_set_hist,
                                                      training_set_ent)

        if split is None:
            return training_set_hist

        (left_data, right_data) = _evaluate_split(training_set, split)

        # Infer the new histograms from the old one and left_data
        hist_right = training_set_hist
        hist_left = [0]*self.n_classes

        for j in left_data[1]:
            hist_left[j] += 1
            hist_right[j] -= 1

        # Recursively train children
        left_node = self._build_extra_tree_rec(left_data, hist_left,
                                               left_ent)
        right_node = self._build_extra_tree_rec(right_data, hist_right,
                                                right_ent)

        return Node(split, left_node, right_node)

    def __output_check(self, outputs):
        # All classes are always integers (this works for booleans too)
        assert all(isinstance(output, int) for output in outputs), (
            "All class labels should be int or bool")

    def _init_build(self, training_set):
        """ Initialise building before calling recursive `build` """
        # Some checks
        attributes, outputs = training_set
        dims = set(len(sample) for sample in attributes)
        assert len(dims) == 1, "Inconsistent attribute sizes"
        assert len(attributes) == len(outputs)

        # We know that classes/attributes are consistent by the check before
        n_attributes = len(attributes[0])
        n_classes = max(outputs) + 1

        self.__output_check(outputs)

        # Since n_classes is the maximum of outputs, we only need to check if
        # they are strictly positive
        assert all(output >= 0 for output in outputs) and min(outputs) == 0, (
            "Class labels should start at 0 and be strictly positive")

        # The default k_values is sqrt(n_attributes) for classification
        # in the original article
        self.k_value = self.k_value or round(math.sqrt(n_attributes))
        self.n_classes = n_classes

    def _score_split(self, subset, split, subset_hist, subset_ent):
        """ Calculate information gain of a potential split node """
        left_out = _evaluate_oneside(subset, split)
        n_classes = self.n_classes

        # We can infer the right histogram from the left one, as the total of
        # the two should is the histogram of the subset
        hist_right = list(subset_hist)
        hist_left = [0]*n_classes

        left_count = 0
        for j in left_out:
            hist_left[j] += 1
            hist_right[j] -= 1
            left_count += 1

        left_ratio = left_count / len(subset[1])

        # If all datapoints are assigned to one side, discard this candidate
        if not left_ratio:
            return -MAXENTROPY, None, None

        criterion = self.criterion

        left_ent = criterion(hist_left)
        right_ent = criterion(hist_right)

        information_gain = (subset_ent
                            - left_ratio*left_ent
                            - (1-left_ratio)*right_ent)

        return information_gain, left_ent, right_ent


class ExtraTreeRegressor(ExtraTreeClassifier):
    """ ExtraTree for regression """
    def __init__(self, k_value=None, n_min=2):
        super(ExtraTreeRegressor, self).__init__(k_value, n_min)
        self.criterion = _variance

    def predict_proba(self, samples):
        """ Regressors have no soft prediction """
        raise NotImplementedError("No probabilistic prediction for regressor")

    def predict(self, samples):
        """ Return the leaf mean value for each sample """
        assert self._fitted, "Tree has not been fitted, call fit() first"

        return [_evaluate_rec(self.root_node, sample) for sample in samples]

    def _make_leaf(self, training_set):
        """ Create a leaf node from available data """
        return _mean(training_set[1])

    def __output_check(self, outputs):
        assert all(isinstance(output, (float, int)) for output in outputs), (
            "All class labels should be floats or ints")


class ExtraForest(object):
    """ Ensemble of ExtraTrees """

    def __init__(self, n_trees=10, **clf_args):
        super(ExtraForest, self).__init__()
        self.n_trees = n_trees
        self.trees = []
        self.clf_args = clf_args
        self._is_classifier = False

    def fit(self, training_set):
        """ Fit each tree in the ensemble (multi-threaded) """
        self.trees = []

        # Classification problem if _all_ outputs are int/bool type, else
        # regression
        is_classifier = all(isinstance(val, int) for val in training_set[1])
        self._is_classifier = is_classifier
        base_classifier = (ExtraTreeClassifier if is_classifier
                           else ExtraTreeRegressor)

        for _ in range(self.n_trees):
            tree = base_classifier(**self.clf_args)
            tree.fit(training_set)
            self.trees.append(tree)

    def predict(self, test_attributes):
        """ Voted, hard-predict the class/value of the `samples` """
        votes = [tree.predict(test_attributes) for tree in self.trees]
        votes_per_sample = list(zip(*votes))
        if not self._is_classifier:
            vote_fun = _mean
        else:
            vote_fun = _majority

        return [vote_fun(sample_votes) for sample_votes in votes_per_sample]

    def __repr__(self):
        fitted_str = "not fitted" if not self.trees else "fitted"
        return "<Forest (%d trees), %s>" % (self.n_trees, fitted_str)
