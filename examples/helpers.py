TRAIN_FILE = './docs/mnist/mnist_train.csv'
TEST_FILE = './docs/mnist/mnist_test.csv'


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


def score(classifier, test_sample):
    """ Accuracy on test set.  """
    predictions = classifier.predict(test_sample[0])
    correct = [pred == true
               for pred, true in zip(predictions, test_sample[1])]

    return sum(correct) / len(correct)
