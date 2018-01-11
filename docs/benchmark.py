import time
import sys
from src.extratrees import Dataset, ExtraForest

""" 
MNIST CSV set from:
https://pjreddie.com/media/files/mnist_train.csv
"""

from pathlib import Path

train_file = Path('./docs/mnist/mnist_train.csv')
test_file = Path('./docs/mnist/mnist_test.csv')

def load_data(file_, n_points=None):
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
        if idx % 10000 == 0:
            print("%5d/%d" % (idx, n_lines))
        split_ = line.split(',')
        labels.append(int(split_[0]))
        attributes.append([int(j) for j in split_[1:]])
    return Dataset(attributes, labels)


def score(classifier, test_sample):
    predictions = classifier.predict(test_sample.attributes)
    correct = [pred == true
               for pred, true in zip(predictions, test_sample.outputs)]

    return sum(correct) / len(correct)

def bench(n_training, n_trees):
    print('-'*20)
    print("Going to train %d trees" % n_trees)
    forest = ExtraForest(n_trees=n_trees, n_min=10)
    print("Loading training data")
    train_set = load_data(train_file, n_points=n_training)

    print("Training...")
    t0 = time.time()
    forest.fit(train_set)
    t_train = time.time() - t0
    print("Trained in %.3fs" % t_train)

    print("Loading test data")
    test_set = load_data(test_file)
    print("Testing")
    t0 = time.time()
    test_score = score(forest, test_set)
    t_test = time.time() - t0
    print("Evaluated in %.3fs" % t_test)
    print("Score: %.3f" % test_score)

if __name__ == '__main__':
    try:
        n_trees = int(sys.argv[2])
        n_training = int(sys.argv[1])
    except (IndexError, ValueError):
        print("Usage: benchmark_mnist.py <n_training> <n_trees>")
        sys.exit(1)

    bench(n_training, n_trees)
