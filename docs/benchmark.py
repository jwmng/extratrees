import time
import sys
from pathlib import Path

from src.extratrees import ExtraForest


TRAIN_FILE = Path('./docs/mnist/mnist_train.csv')
TEST_FILE = Path('./docs/mnist/mnist_test.csv')


def load_data(file_, n_points=None):
    """
    Load CSV data from `file` and return the first `n` points as Dataset

    The CSV file should have the first column as labels and the rest as data.
    All data must be integer.
    Works well with the MNIST set from:
    https://pjreddie.com/media/files/mnist_train.csv

    Args:
        file_ (pathlib.Path): File to load
        n_points (int): Number of datapoints to load. Loads all if not `None`.
    """
    try:
        lines = file_.read_text().splitlines()[:n_points]
    except FileNotFoundError:
        print("The MNIST datafiles could not be found. Please see readme.md")

    if n_points:
        lines = lines[:n_points]

    labels = []
    attributes = []
    n_lines = len(lines)
    for idx, line in enumerate(lines):
        if idx % (n_lines/10) == 0:
            print("%5d/%d" % (idx, n_lines))
        split_ = line.split(',')
        labels.append(int(split_[0]))
        attributes.append([int(j) for j in split_[1:]])
    return (attributes, labels)


def score(classifier, test_sample):
    """ Accuracy on test set.  """
    predictions = classifier.predict(test_sample[0])
    correct = [pred == true
               for pred, true in zip(predictions, test_sample[1])]

    return sum(correct) / len(correct)


def bench(train_data, test_data, n_trees):
    """
    Benchmark Extratrees

    Trains a ExtraForest with `n_trees` trees and returns training time,
    evaluation time and testing accuracy.

    Args:
        train_data (extratrees.Datset): Train set
        test_data (extratrees.Datset): Test set
        n_trees (int): Number of trees
    """

    print('-'*20)
    print("Going to train %d trees" % n_trees)
    forest = ExtraForest(n_trees=n_trees, n_min=10, criterion="entropy")

    print("Training...")
    t_start = time.time()
    forest.fit(train_data)
    t_train = time.time() - t_start
    print("Trained in %.3fs" % t_train)

    print("Testing")
    t_start = time.time()
    test_score = score(forest, test_data)
    t_test = time.time() - t_start
    print("Evaluated in %.3fs" % t_test)
    print("Score: %.3f" % test_score)


if __name__ == '__main__':
    try:
        N_TREES = int(sys.argv[2])
        N_TRAINING = int(sys.argv[1])
    except (IndexError, ValueError):
        print("Usage: benchmark_mnist.py <n_training> <n_trees>")
        sys.exit(1)

    print("Loading training data")
    TRAIN_SET = load_data(TRAIN_FILE, n_points=N_TRAINING)

    print("Loading test data")
    TEST_SET = load_data(TEST_FILE)
    bench(TRAIN_SET, TEST_SET, N_TREES)
