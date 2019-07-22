# -*- coding: utf-8 -*-

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
from ninolearn.learn.models.encoderDecoder import EncoderDecoder
from ninolearn.pathes import ed_model_dir
from ninolearn.utils import print_header
from ninolearn.learn.evaluation import correlation
from ninolearn.plot.evaluation import plot_seasonal_skill

plt.close("all")

#%%free memory from old sessions
K.clear_session()
decades_arr = [60, 70, 80, 90, 100, 110]
n_decades = len(decades_arr)
# =============================================================================
# Animation
# =============================================================================

def animation_ed(true, pred, nino,  nino_pred, time):
    fig, ax = plt.subplots(3, 1, figsize=(6,7), squeeze=False)

    vmin = -3
    vmax = 3

    true_im = ax[0,0].imshow(true[0], origin='lower', vmin=vmin, vmax=vmax, cmap=plt.cm.bwr)
    pred_im = ax[1,0].imshow(pred[0], origin='lower', vmin=vmin, vmax=vmax, cmap=plt.cm.bwr)
    title = ax[0,0].set_title('')

    ax[2,0].plot(time, nino)
    ax[2,0].plot(time, nino_pred)
    ax[2,0].set_ylim(-3,3)
    ax[2,0].set_xlim(time[0], time[-1])

    vline = ax[2,0].plot([time[0], time[0]], [-10,10], color='k')


    def update(data):
        true_im.set_data(data[0])
        pred_im.set_data(data[1])
        title_str = np.datetime_as_string(data[0].time.values)[:10]
        title.set_text(title_str)

        vline[0].set_data([data[2], data[2]],[-10,10])

    def data_gen():
        k=0
        kmax = len(label)
        while k<kmax:

            yield true.loc[time[k]], pred.loc[time[k]], time[k]
            k+=1

    ani = animation.FuncAnimation(fig, update, data_gen, interval=100)
    return ani

#%% =============================================================================
# Data
# =============================================================================
reader = data_reader(startdate='1959-11', enddate='2017-12')
sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom').rolling(time=3).mean()[2:]
oni = reader.read_csv('oni')[2:]

# select
feature = sst.copy(deep=True)
label = sst.copy(deep=True)

# preprocess data
feature_unscaled = feature.values.reshape(feature.shape[0],-1)
label_unscaled = label.values.reshape(label.shape[0],-1)

scaler_f = StandardScaler()
Xorg = scaler_f.fit_transform(feature_unscaled)

scaler_l = StandardScaler()
yorg = scaler_l.fit_transform(label_unscaled)

Xall = np.nan_to_num(Xorg)
yall = np.nan_to_num(yorg)

shift = 3
lead = 9
print_header(f'Lead time: {lead} months')

y = yall[lead+shift:]
X = Xall[:-lead-shift]

timey = oni.index[lead+shift:]

y_nino = oni[lead+shift:]

pred_full_oni = np.array([])
true_oni = np.array([])
timeytrue = pd.DatetimeIndex([])

pred_da_full = xr.zeros_like(label[lead+shift:,:,:])

for j in range(n_decades):
    decade = decades_arr[j]

    print_header(f'Predict: {1902+decade}-01-01 till {1911+decade}-12-01')

    # get test data
    test_indeces = test_indeces = (timey>=f'{1902+decade}-01-01') & (timey<=f'{1911+decade}-12-01')
    testX, testy, testtimey = X[test_indeces,:], y[test_indeces,:], timey[test_indeces]
    testnino  = y_nino[test_indeces]

    # load model corresponding to the test data
    model = EncoderDecoder()
    dir_name = f'ed_ensemble_decade{decade}_lead{lead}'
    model.load(location=ed_model_dir, dir_name=dir_name)

    # make prediction
    pred, _ = model.predict(testX)

    # reshape data into an xarray data array (da)
    pred_da = xr.zeros_like(label[lead+shift:,:,:][test_indeces])
    pred_da.values = scaler_l.inverse_transform(pred).reshape((testX.shape[0], label.shape[1], label.shape[2]))

    # fill full forecast array
    timeytrue = timeytrue.append(testtimey)
    pred_da_full.values[test_indeces] = pred_da.values

    # calculate the ONI
    pred_oni = pred_da.loc[dict(lat=slice(-5, 5), lon=slice(190, 240))].mean(dim="lat").mean(dim='lon')

    # make the full time series
    pred_full_oni = np.append(pred_full_oni, pred_oni)
    true_oni = np.append(true_oni, testnino)


#%%
ani = animation_ed(label[lead+shift:], pred_da_full, true_oni, pred_full_oni, timeytrue)