# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras.backend as K

from ninolearn.plot.evaluation  import plot_correlation
from ninolearn.plot.prediction import plot_prediction
from ninolearn.learn.evaluation import rmse
from ninolearn.learn.dem import DEM
from ninolearn.utils import print_header
from ninolearn.pathes import modeldir

from data_pipeline import pipeline

K.clear_session()
#%% =============================================================================
# Deep ensemble
# =============================================================================
lead_time = 10
X, y, timey, yp = pipeline(lead_time, return_persistance=True)
#%%
decades = [80, 90, 100, 110]

for decade in decades:
    K.clear_session()
    print_header(f'Test period: {1902+decade}-01-01 till {1911+decade}-12-01')

    test_indeces = (timey>=f'{1902+decade}-01-01') & (timey<=f'{1911+decade}-12-01')
    train_indeces = np.invert(test_indeces)

    trainX, trainy = X[train_indeces,:], y[train_indeces]

    model = DEM()

    model.set_parameters(layers=1, dropout=0.2, noise=0.2, l1_hidden=0.0,
                l2_hidden=0.2, l1_mu=0., l2_mu=0.2, l1_sigma=0.0, l2_sigma=0.2,
                lr=0.001, batch_size=1, epochs=500, n_segments=5,
                n_members_segment=1, patience=30, verbose=0, std=True)

    model.fit(trainX, trainy)

    if model.std:
        ens_dir=f'ensemble_decade{decade}_lead{lead_time}'

    else:
        ens_dir=f'ensemble_decade{decade}_lead{lead_time}'

    model.save(location=modeldir, dir_name=ens_dir)

    del model

#%% just for testing the loading function delete and load the model
i=0
pred_mean_full = np.array([])
pred_std_full = np.array([])

decadal_corr = np.zeros(len(decades))
decadal_rmse = np.zeros(len(decades))
decadal_nll = np.zeros(len(decades))

for decade in decades:
    K.clear_session()
    print_header(f'Predict: {1902+decade}-01-01 till {1911+decade}-12-01')

    ens_dir=f'ensemble_decade{decade}_lead{lead_time}'
    model = DEM()
    model.load(location=modeldir, dir_name=ens_dir)

    test_indeces = (timey>=f'{1902+decade}-01-01') & (timey<=f'{1911+decade}-12-01')
    testX, testy = X[test_indeces,:], y[test_indeces]
    pred_mean, pred_std = model.predict(testX)

    pred_mean_full = np.append(pred_mean_full, pred_mean)
    pred_std_full = np.append(pred_std_full, pred_std)

    decadal_nll[i] = model.evaluate(testy, pred_mean, pred_std)
    decadal_rmse[i] = round(rmse(testy, pred_mean),2)
    decadal_corr[i] = np.corrcoef(testy, pred_mean)[0,1]
    i+=1


#%% Predictions
plt.close("all")
plt.subplots(figsize=(15,3.5))
plt.axhspan(-0.5,
            -6,
            facecolor='blue',
            alpha=0.1,zorder=0)

plt.axhspan(0.5,
            6,
            facecolor='red',
            alpha=0.1,zorder=0)

plt.xlim(timey[0],timey[-1])
plt.ylim(-3,3)

# test
plot_prediction(timey, pred_mean_full, std=pred_std_full, facecolor='royalblue', line_color='navy')

# observation
plt.plot(timey, y, "k")

pred_rmse = round(rmse(y, pred_mean_full),2)
plt.title(f"Lead time: {lead_time} month, RMSE (of mean): {pred_rmse}")
plt.grid()

# plot explained variance
# minus one month to center season around central month
plot_correlation(y, pred_mean_full, timey - pd.tseries.offsets.MonthBegin(1))


# Error distribution
plt.subplots()
plt.title("Error distribution")
error = pred_mean_full - y

plt.hist(error, bins=16)

#decadal scores
decades_plot = ['80s', '90s', '00s', '10s']

fig, ax = plt.subplots(1,3, figsize=(8, 4))
ax[0].set_title("NLL")
ax[0].bar(decades_plot, decadal_nll)
ax[0].set_ylim(-0.5, 0.5)

ax[1].set_title("correlation")
ax[1].bar(decades_plot, decadal_corr)
ax[1].set_ylim(-0.1,1)

ax[2].set_title("rmse")
ax[2].bar(decades_plot, decadal_rmse)
ax[1].set_ylim(0.,1)

