from math import pi, sin, cos
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.extratrees import (Dataset, ExtraTree, ExtraForest)


def colorize_class(cls):
    """ Map 0..1 to (1, 0, 0)..(0, 0, 1) """
    return (cls, 0, 1-cls)


def create_dataset():
    """ Create spiral dataset """
    radius = [r/100 for r in range(100)]
    angle = [a/(25/pi) for a in range(100)]

    x1 = [r*cos(a) for r, a in zip(radius, angle)]
    y1 = [r*sin(a) for r, a in zip(radius, angle)]
    x2 = [r*cos(a-pi) for r, a in zip(radius, angle)]
    y2 = [r*sin(a-pi) for r, a in zip(radius, angle)]

    attributes = list(zip(x1+x2, y1+y2))
    outputs = [0.0]*len(x1) + [1.0]*len(x2)
    data = Dataset(attributes, outputs)
    return data


def show_decision_grid(ax, classifier, gridsize):
    for x_i in range(gridsize):
        x = -1+2*x_i/gridsize
        for y_i in range(gridsize):
            y = -1+2*y_i/gridsize
            grid_pred = classifier.predict([[x, y]])
            col = (grid_pred[0]), 0, (1-grid_pred[0])
            ax.add_patch(patches.Rectangle((x, y),
                                           (2/gridsize),
                                           (2/gridsize),
                                           facecolor=col))

# Create data
data = create_dataset()

# Train tree
tree = ExtraTree(n_min=5)
tree.fit(data)

# Train forest
forest = ExtraForest(n_trees=10, n_min=5)
forest.fit(data)

# Predict
pred_tree = tree.predict(data.attributes)
pred_forest = forest.predict(data.attributes)

fig, axes = plt.subplots(2, 3)

# Plot data
axes[0][0].set_title('True')
for coords, cls in zip(*data):
    axes[0][0].plot(*coords, marker='o', color=colorize_class(cls))


# Plot tree result
axes[0][1].set_title("Single tree\n (min_samples=5, k=*)")
for idx, cls in enumerate(pred_tree):
    axes[0][1].plot(*data.attributes[idx], marker='o',
                    color=colorize_class(cls))

# Plot forest result
axes[0][2].set_title("10 trees\n (min_samples=5, k=*)")
for idx, cls in enumerate(pred_forest):
    axes[0][2].plot(*data.attributes[idx], marker='o',
                    color=colorize_class(cls))


show_decision_grid(axes[1][1], tree, 50)
show_decision_grid(axes[1][2], forest, 50)
axes[1][1].set_title('Decision boundaries')
axes[1][2].set_title('Decision boundaries')

for row in axes:
    for ax in row:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()
