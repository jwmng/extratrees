import time
import sys

from src.extratrees import ExtraForest
from docs.helpers import load_data, score, TRAIN_FILE, TEST_FILE


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

    print('-'*20, "This implementation")
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
    TEST_SET = load_data(TEST_FILE, n_points=1000)
    bench(TRAIN_SET, TEST_SET, N_TREES)
