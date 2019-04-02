import matplotlib.pyplot as plt
import numpy as np

from ninolearn.learn.evaluation import explained_variance
from ninolearn.utils import scale


# scatter
def plot_correlations(y, pred, time):
    fig, ax = plt.subplots(4, 3, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.5)
    pos = np.argwhere(np.zeros((4, 3)) == 0)

    rsq = explained_variance(y, pred, time)

    for i in range(0, 12):
        month = (time.month == i+1)
        y_plot = scale(y[month])
        pred_plot = scale(pred[month])

        ax[pos[i, 0], pos[i, 1]].scatter(y_plot, pred_plot)
        ax[pos[i, 0], pos[i, 1]].set_xlim([-3, 3])
        ax[pos[i, 0], pos[i, 1]].set_ylim([-3, 3])

        ax[pos[i, 0], pos[i, 1]].set_title(f"month: {i}, r$^2$:{rsq[i]}")


def plot_explained_variance(y, pred, time):
    """
    make a bar plot of the explained varince between y and the prediction
    """
    m = np.arange(1, 13)
    fig, ax = plt.subplots()

    rsq = explained_variance(y, pred, time)

    ax.set_ylim(0, 1)
    ax.bar(m, rsq)
    ax.set_xticks(m)
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J',
                        'J', 'A', 'S', 'O', 'N', 'D'])
    ax.set_xlabel("month")
    ax.set_ylabel(f"$r^2$")
