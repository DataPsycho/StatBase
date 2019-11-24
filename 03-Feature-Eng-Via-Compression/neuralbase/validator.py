from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_regions(features, targets, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(targets))])
    # plot the decision surface
    # create a min max for x and y axis
    x1_min, x1_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    x2_min, x2_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    # create a cartesian combination of x and y values
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    # predict for each grid combination
    grid_prediction = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # reshape to grid pair
    grid_prediction = grid_prediction.reshape(xx1.shape)
    # make the grid contour plot
    plt.contourf(xx1, xx2, grid_prediction, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot actual values
    for idx, cl in enumerate(np.unique(grid_prediction)):
        plt.scatter(
            x=features[targets == cl, 0],
            y=features[targets == cl, 1],
            alpha=0.6,
            c=colors[idx],
            edgecolor='black',
            marker=markers[idx],
            label=cl
        )


def plot_decision_regions_combined(features, targets, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(targets))])

    # plot the decision surface
    x1_min, x1_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    x2_min, x2_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    # create an space for all combination
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    result_space = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    result_space = result_space.reshape(xx1.shape)
    plt.contourf(xx1, xx2, result_space, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot trained values in a loop by class type
    for idx, cl in enumerate(np.unique(targets)):
        plt.scatter(x=features[targets == cl, 0],
                    y=features[targets == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # If test and train data combined
    # then mark the test data after ward by o
    if test_idx:
        # plot all samples
        x_test = features[test_idx, :]
        plt.scatter(
            x_test[:, 0],
            x_test[:, 1],
            c='',
            edgecolor='black',
            alpha=1.0,
            linewidth=1,
            marker='o',
            s=100,
            label='test set'
        )
