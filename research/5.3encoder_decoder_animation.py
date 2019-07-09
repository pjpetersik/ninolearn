import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator

import pandas as pd

import keras.backend as K

from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr

from ninolearn.IO.read_post import data_reader
from ninolearn.learn.encoderDecoder import EncoderDecoder
from ninolearn.pathes import ed_model_dir
from ninolearn.utils import print_header
from ninolearn.learn.evaluation import correlation
from ninolearn.plot.evaluation import plot_seasonal_skill




# =============================================================================
# Animation
# =============================================================================
fig, ax = plt.subplots(3, 1, figsize=(6,7), squeeze=False)

vmin = -3
vmax = 3
true = label[lead+shift:][test_indeces]
pred = predy_decoded

true_im = ax[0,0].imshow(true[0], origin='lower', vmin=vmin, vmax=vmax, cmap=plt.cm.bwr)
pred_im = ax[1,0].imshow(pred[0], origin='lower', vmin=vmin, vmax=vmax, cmap=plt.cm.bwr)
title = ax[0,0].set_title('')


ax[2,0].plot(nino34_pred.time, nino34.values[lead+shift:][test_indeces])
ax[2,0].plot(nino34_pred.time, nino34_pred.values)
ax[2,0].set_ylim(-3,3)
ax[2,0].set_xlim(nino34_pred.time.values[0], nino34_pred.time[-1].values)

vline = ax[2,0].plot([nino34_pred.time.values[0], nino34_pred.time[0].values], [-10,10], color='k')


def update(data):
    true_im.set_data(data[0])
    pred_im.set_data(data[1])
    title_str = np.datetime_as_string(data[0].time.values)[:10]
    title.set_text(title_str)

    vline[0].set_data([data[2], data[2]],[-10,10])

def data_gen():
    k=0
    kmax = len(testy)
    while k<kmax:
        yield true[k], pred[k], nino34_pred.time[k].values
        k+=1

ani = animation.FuncAnimation(fig, update, data_gen, interval=100)
