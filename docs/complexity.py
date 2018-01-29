import time
import numpy as np
import matplotlib.pyplot as plt

from src.extratrees import ExtraForest

TRAIN_FILE = './docs/mnist/mnist_train.csv'


def load_data(file_, n_points=None):
    """
    Load CSV data from `file` and return the first `n` points as Dataset

    The CSV file should have the first column as labels and the rest as data.
    All data must be integer.
    Works well with the MNIST set from:
    https://pjreddie.com/media/files/mnist_train.csv

    Args:
        file_ (str): File path to load
        n_points (int): Number of datapoints to load. Loads all if not `None`.
    """
    try:
        with open(file_) as infile:
            data = infile.read()
    except FileNotFoundError:
        print("The MNIST datafiles could not be found. Please see readme.md")

    lines = data.splitlines()[:n_points]

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


def get_train_time(train_data):
    """
    Benchmark Extratrees

    Trains a ExtraTree trees and returns training time.

    Args:
        train_data (extratrees.Datset): Train set
    """

    forest = ExtraForest(n_trees=10, n_min=10, criterion="entropy")

    t_start = time.time()
    forest.fit(train_data)
    t_train = time.time() - t_start
    return t_train

if __name__ == '__main__':
    N_POINTS = 20
    TRAIN_SET = load_data(TRAIN_FILE, n_points=60000)

    train_times = np.zeros(N_POINTS)
    train_sizes = np.logspace(1, 4.5, N_POINTS)
    for idx, n_train in enumerate(train_sizes):
        train_set = (TRAIN_SET[0][:int(n_train)], TRAIN_SET[1][:int(n_train)])
        train_times[idx] = get_train_time(train_set)
        print(int(n_train), train_times[idx])

    plt.xlabel('n')
    plt.ylabel('time [s]')
    plt.loglog(train_sizes, train_times/train_times[0], label='Algorithm')

    # We want each line to start at train_times[0]
    linear_sc = train_sizes/train_sizes[0]
    quadratic_sc = (train_sizes)**2
    quadratic_sc = quadratic_sc / quadratic_sc[0]
    nlogn_sc = train_sizes*np.log(train_sizes)
    nlogn_sc = nlogn_sc / nlogn_sc[0]

    plt.loglog(train_sizes, linear_sc, label='n')
    plt.loglog(train_sizes, quadratic_sc, label='n^2')
    plt.loglog(train_sizes, nlogn_sc, label='n*logn')
    plt.legend()
    plt.grid(True)
    plt.show()
