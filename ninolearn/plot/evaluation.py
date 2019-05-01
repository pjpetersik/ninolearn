import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from ninolearn.learn.evaluation import explained_variance, correlation
from ninolearn.utils import scale



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
    ax.set_title(f"$r^2 =$ {round(np.corrcoef(y,pred)[0,1]**2, 2)}")

def plot_correlation(y, pred, time):
    """
    make a bar plot of the explained varince between y and the prediction
    """
    m = np.arange(1, 13)
    fig, ax = plt.subplots()

    rsq = correlation(y, pred, time)

    ax.set_ylim(0, 1)
    ax.bar(m, rsq)
    ax.set_xticks(m)
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J',
                        'J', 'A', 'S', 'O', 'N', 'D'])
    ax.set_xlabel("month")
    ax.set_ylabel(f"$r$")
    ax.set_title(f"$r =$ {round(np.corrcoef(y,pred)[0,1], 2)}")

def plot_confMat(y, pred, labels):
    """
    Plot a confusion matrix. Here, the recall is on the diagonal!

    :param y: The baseline.
    :param pred: The prediction.
    :param labels: The names of the classes.
    """
    cm = confusion_matrix(y, pred)#.T
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues,
                   vmin = 1/len(labels), vmax = 0.8)
    ax.figure.colorbar(im, ax=ax,extend='max')
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title='Confusion Matrix',
           xlabel='True label',
           ylabel='Predicted label')

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black" if cm[i, j] > thresh else "black")
    fig.tight_layout()
