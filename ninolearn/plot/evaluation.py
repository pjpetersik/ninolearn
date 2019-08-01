import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.metrics import confusion_matrix

from ninolearn.learn.evaluation import explained_variance, correlation

seismic = plt.cm.get_cmap('seismic', 256)
newcolors = seismic(np.linspace(0, 1, 256))
grey = np.array([192/256, 192/256, 192/256, 1])
newcolors[:1, :] = grey
newcmp = ListedColormap(newcolors)

seas_ticks = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
                        'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']

mon_ticks = ['J', 'F', 'M', 'A', 'M', 'J',
                        'J', 'A', 'S', 'O', 'N', 'D']

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
    ax.set_xticklabels(seas_ticks)
    ax.set_xlabel("Season")
    ax.set_ylabel(f"$r^2$")
    ax.set_title(f"$r^2 =$ {round(np.corrcoef(y,pred)[0,1]**2, 2)}")

def plot_correlation(y, pred, time, title=None):
    """
    make a bar plot of the correlation coeficent between y and the prediction
    """
    m = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(5,2.5))

    r, p = correlation(y, pred, time)

    ax.set_ylim(0, 1)
    ax.bar(m, r)
    ax.set_xticks(m)
    ax.set_xticklabels(seas_ticks)
    ax.set_xlabel("Season")
    ax.set_ylabel(f"Correlation coefficient")
    if title is None:
        ax.set_title(f"$r =$ {round(np.corrcoef(y,pred)[0,1], 2)}")
    else:
         ax.set_title(title)
    plt.tight_layout()

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


def plot_seasonal_skill(lead_time, data, vmin=-1, vmax=1, nlevels=20, cmap=newcmp, extend='min'):
    fig, ax = plt.subplots(figsize=(5,3.5))
    m = np.arange(1,13)

    levels = np.linspace(vmin, vmax, nlevels+1)
    C = ax.contourf(m,lead_time, data, levels=levels,
                 vmin=vmin, vmax=vmax,
                 cmap=cmap, extend=extend)

    ax.set_xticks(m)
    ax.set_xticklabels(seas_ticks, rotation='vertical')
    ax.set_xlabel('Target Season')
    ax.set_yticks(lead_time)
    ax.set_yticklabels(lead_time)
    ax.set_ylabel('Lead Time [Month]')
    plt.colorbar(C, ticks=np.arange(vmin,vmax+0.1,0.2))
    plt.tight_layout()
