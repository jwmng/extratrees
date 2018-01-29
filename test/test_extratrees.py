import math
import random
import unittest
from src.extratrees import (_entropy, _histogram, _mean, 
                            _evaluate_split, ExtraTreeClassifier, Node,
                            _pick_random_split, ExtraForest)

FOURTHS = ([[0], [1], [1], [1],
            [1], [0], [0], [0]],
            [0, 0, 0, 0, 1, 1, 1, 1])


class TestModuleFunctions(unittest.TestCase):
    def test_histogram(self):
        self.assertEqual(_histogram([0, 0, 0, 1, 1, 2]), [3/6, 2/6, 1/6])
        self.assertEqual(_histogram([0, 0, 0]), [1])
        self.assertEqual(_histogram([1, 1, 1]), [0, 1])

    def test_entropy(self):
        self.assertEqual(_entropy([1, 1]), 0)
        self.assertEqual(_entropy([0, 1]), -math.log(0.5))
        self.assertAlmostEqual(_entropy([0, 1, 1, 1]), 0.56, 2)
        self.assertAlmostEqual(_entropy([0, 1, 1]), 0.64, 2)
        self.assertEqual(_entropy([]), -1e10)

    def test_mean(self):
        self.assertEqual(_mean([1]), 1)
        self.assertEqual(_mean([0, 1]), 0.5)
        self.assertEqual(_mean([1, 1]), 1)
        self.assertEqual(_mean([1, 2, 3, 4]), 2.5)

    def test_pick_random(self):
        random.seed(0)
        data = ([[0, 0], [0, 1], [0, 1]], [1, 1, 1])
        split = _pick_random_split(data, 1)

        # Types
        self.assertIsInstance(split, Split)
        self.assertEqual(split.attribute, 1)

        # Seed is fixed here
        self.assertAlmostEqual(split.cutoff, 0.8444, 4)

        # Only one value possible for attribute 1
        self.assertEqual(_pick_random_split(data, 0).cutoff, 0)

    def test_evaluate_split(self):
        random.seed(0)
        data = ([[0, 0], [0, 1], [0, 1]], [1, 1, 1])
        split = _pick_random_split(data, 1)
        left, right = _evaluate_split(data, split)
        self.assertEqual(right, (([0, 0],), (1,)))
        self.assertEqual(left, (([0, 1], [0, 1]), (1, 1)))


class TestExtraTreeObject(unittest.TestCase):
    def setUp(self):
        self.data = ([[0, 0], [0, 1], [0, 1]], [0, 1, 1])
        self.tree = ExtraTree(4, 1)

        self.fourths = FOURTHS
        self.fourths_split = Split(0, 0.5)

    def test_init(self):
        ext = ExtraTree(2, 3)
        self.assertEqual(ext.k_value, 2)
        self.assertEqual(ext.n_min, 3)
        self.assertEqual(ext._fitted, False)
        self.assertEqual(ext.is_classifier(), False)

    def test_score_classification(self):
        self.tree._is_classifier = True

        # This is a perfect split for self.data
        split = Split(1, 0.5)
        self.assertAlmostEqual(self.tree._score_split(self.data, split),
                               0.64, 2)

        # This is a worthless split
        split2 = Split(0, 0.5)
        self.assertEqual(self.tree._score_split(self.data, split2), 0.0)

        # Hand-calculated example from:
        # http://www.bogotobogo.com/python/scikit-learn/scikt_machine_learning_
        # Decision_Tree_Learning_Informatioin_Gain_IG_Impurity_Entropy_Gini_Cla
        # ssification_Error.php
        self.assertAlmostEqual(self.tree._score_split(self.fourths,
                                               self.fourths_split),
                               0.13, 2)

    def test_score_regression(self):
        split = Split(1, 0.5)
        self.assertEqual(self.tree._score_split(self.data, split), 1)

        split2 = Split(0, 0.5)
        self.assertEqual(self.tree._score_split(self.data, split2), 0.0)

        self.assertAlmostEqual(self.tree._score_split(self.fourths,
                                               self.fourths_split),
                               0.125, 2)

    def test_stop_split(self):
        # Empty set
        data = ([], [])
        self.assertTrue(self.tree._stop_split(data))

        # Constant outputs
        data = ([[0], [1]], [1, 1])
        self.assertTrue(self.tree._stop_split(data))

        # Fully constant attributes
        data = ([[0, 1], [0, 1]], [0, 1])
        self.assertTrue(self.tree._stop_split(data))

        # Proper sets
        self.assertFalse(self.tree._stop_split(self.data))
        self.assertFalse(self.tree._stop_split(self.fourths))

    def test_fit(self):
        tree = ExtraTree(k_value=4)
        tree.fit(self.fourths)

        # Fourths will be split anywhere between 0-1, and then only have
        # uniform attributes
        self.assertEqual(tree.root_node.split.attribute, 0)
        self.assertTrue(0 < tree.root_node.split.cutoff < 1)
        self.assertEqual(tree.root_node.left, [0.75, 0.25])
        self.assertEqual(tree.root_node.right, [0.25, 0.75])

        # This always results in a pure split
        tree.fit(self.data)
        self.assertTrue(0 < tree.root_node.split.cutoff < 1)
        self.assertEqual(tree.root_node.left, [0.0, 1.0])
        self.assertEqual(tree.root_node.right, [1.0, 0.0])

        # Dataset with two-level splits
        xval = [[0, 0, 1, 1],
                [0, 1, 1, 0],
                [1, 1, 0, 0],
                [1, 0, 0, 1]]
        yval = [0, 1, 0, 1]
        data = (xval, yval)
        random.seed(0)
        tree = ExtraTree()
        tree.fit(data)

        # We must have 2 levels here
        self.assertIsInstance(tree.root_node.left, Node)
        self.assertIsInstance(tree.root_node.right, Node)

        # The leaf-leaves are always pure
        self.assertEqual(tree.root_node.left.left, [0.0, 0.0])
        self.assertEqual(tree.root_node.left.right.left, [1.0, 0.0])
        self.assertEqual(tree.root_node.right.left, [1.0, 0.0])
        self.assertEqual(tree.root_node.right.right, [0.0, 1.0])

    def test_build_tree_regression(self):
        xdata = ([0], [0], [0], [0], [1], [1], [1], [1])
        ydata = (2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3)
        tree = ExtraTree()
        tree.fit((xdata, ydata))
        self.assertEqual(tree.root_node.split.attribute, 0)
        self.assertTrue(0 < tree.root_node.split.cutoff < 1)
        self.assertAlmostEqual(tree.root_node.left, 3.15)
        self.assertAlmostEqual(tree.root_node.right, 2.15)

    def test_evaluate_tree(self):
        left_node = Node(Split(0, 0.5), left=[0.9, 0.1], right=[0.8, 0.2])
        right_node = Node(Split(0, 2.5), left=[0.3, 0.7], right=[0.4, 0.6])
        tree = ExtraTree()
        tree.root_node = Node(Split(0, 1.5), left=left_node, right=right_node)
        tree._fitted = True

        xdata = [[0], [1], [2], [3]]
        pred = tree.predict_proba(xdata)

        # x[0] is < 2.5 (left) and < 1.5 (left) => [0.1, 0.9]
        self.assertEqual(pred[0], [0.4, 0.6])

        # x[1] is < 2.5 (left) and < 1.5 (left) => 0.9
        self.assertEqual(pred[1], [0.4, 0.6])

        # x[2] is < 2.5 (right) and > 1.5 (left) => 0.4
        self.assertEqual(pred[2], [0.9, 0.1])

        # x[3] is > 2.5 (left) and < 3.5 (right) => 0.6
        self.assertEqual(pred[3], [0.9, 0.1])

        pred = tree.predict(xdata)
        self.assertEqual(pred, [1, 1, 0, 0])

        # Regression
        tree = ExtraTree()
        tree.root_node = Node(Split(0, 1.5), left=0.1, right=0.8)
        tree._fitted = True
        pred = tree.predict([[0], [3]])
        self.assertEqual(pred, [0.8, 0.1])

    def test_empty_fit(self):
        tree = ExtraTree()
        # This makes the root node a leaf node, which causes problems and
        # therefore is not allowed
        with self.assertRaises(ValueError):
            tree.fit(([[1, 1]], [0]))



class TestForest(unittest.TestCase):
    def setUp(self):
        self.fourths = FOURTHS

    def test_init_forest(self):
        forest = ExtraForest(n_trees=5, k_value=3, n_min=1)
        self.assertEqual(forest.n_trees, 5)
        self.assertEqual(forest.k_value, 3)
        self.assertEqual(forest.n_min, 1)
        self.assertFalse(forest._is_classifier)
        self.assertEqual(forest.trees, [])

    def test_fit(self):
        forest = ExtraForest(n_trees=3)
        forest.fit(self.fourths)

        self.assertTrue(forest._is_classifier)
        self.assertTrue(len(forest.trees), 3)

        for tree in forest.trees:
            self.assertIsInstance(tree, ExtraTree)
            self.assertTrue(tree._is_classifier)

    def test_predict(self):
        random.seed(0)
        forest = ExtraForest(n_trees=3, n_min=1)
        forest.fit(self.fourths)
        pred = forest.predict(self.fourths[0])

        # For all trees, the first split will be somewhere inbetween the single
        # attributes of `fourths[0`. There will be no second split as
        # the attributes are uniform after the first split.
        for tree in forest.trees:
            self.assertEqual(tree.root_node.left, [0.75, 0.25])
            self.assertEqual(tree.root_node.right, [0.25, 0.75])

        self.assertEqual(pred, [1, 0, 0, 0, 0, 1, 1, 1])

    def test_predict_forest_classify(self):
        random.seed(0)
        forest = ExtraForest(n_trees=3, n_min=1)
        forest.fit(self.fourths)
        pred = forest.predict(self.fourths[0])

        for tree in forest.trees:
            self.assertEqual(tree.root_node.left, [0.75, 0.25])
            self.assertEqual(tree.root_node.right, [0.25, 0.75])

        self.assertEqual(pred, [1, 0, 0, 0, 0, 1, 1, 1])

    def test_predict_forest_regression(self):
        xdata = ([0], [0], [0], [0], [1], [1], [1], [1])
        ydata = (2.0, 2.1, 2.2, 2.3, 3.0, 3.1, 3.2, 3.3)
        forest = ExtraForest()
        forest.fit((xdata, ydata))
        pred = forest.predict(xdata)

        # These are the means of the xdata classes, since after the first split
        # the attributes are uniform
        for zval in pred[:4]:
            self.assertAlmostEqual(zval, 2.15)

        for zval in pred[4:]:
            self.assertAlmostEqual(zval, 3.15)

    def test_repr_no_fit(self):
        forest = ExtraForest()
        self.assertEqual(str(forest), "<Forest (10 trees), not fitted>")
