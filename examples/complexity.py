import time
import numpy as np
import matplotlib.pyplot as plt

from src.extratrees import ExtraForest
from docs.helpers import TRAIN_FILE, load_data


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
    t_train = (time.time() - t_start) / 10
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
    plt.loglog(train_sizes, train_times, label='Time')
    plt.grid(True)
    plt.legend()
    plt.show()
