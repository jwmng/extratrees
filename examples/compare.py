""" compare.py - Compare training time of extratrees and sklearn """

import time
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier

from src.extratrees import ExtraForest
from docs.helpers import load_data, TRAIN_FILE, TEST_FILE


if __name__ == '__main__':
    TRAIN_SET = load_data(TRAIN_FILE)
    TEST_SET = load_data(TEST_FILE)

    this_clf = ExtraForest(n_trees=1, n_min=10, criterion="gini")
    skl_clf = ExtraTreesClassifier(criterion="gini", min_samples_split=10)


    sizes = (100, 1000, 2000, 5000, 10000, 20000, 50000)
    times_this = [0]*len(sizes)
    times_skl = [0]*len(sizes)

    for idx, size in enumerate(sizes):

        # This clf
        t0 = time.time()
        this_clf.fit((TRAIN_SET[0][:size], TRAIN_SET[1][:size]))
        times_this[idx] = time.time()-t0

        # Sklearn
        t0 = time.time()
        skl_clf.fit(TRAIN_SET[0][:size], TRAIN_SET[1][:size])
        times_skl[idx] = time.time() - t0

    plt.semilogx(sizes, times_this, label="extratrees.py")
    plt.semilogx(sizes, times_skl, label="sklearn")
    plt.xlabel('Number of training points')
    plt.ylabel('Training time [s]')
    plt.legend()
    plt.show()
