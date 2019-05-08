# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from keras import backend as K

from ninolearn.learn.dem import DEM
from ninolearn.pathes import modeldir
from ninolearn.learn.evaluation import rmse, correlation, rmse_mon
from ninolearn.plot.evaluation import plot_seasonal_skill

from data_pipeline import pipeline

#%% =============================================================================
#  process data
# =============================================================================
decades = [80, 90, 100, 110]

lead_time_arr = np.array([0, 3, 6, 9])

n_pred = len(lead_time_arr)
all_season_corr = np.zeros(n_pred)
all_season_rmse = np.zeros(n_pred)
all_season_corr_pres = np.zeros(n_pred)
all_season_rmse_pres = np.zeros(n_pred)
all_season_nll = np.zeros(n_pred)

seas_corr = np.zeros((12, n_pred))
seas_corr_pers = np.zeros((12, n_pred))
seas_rmse = np.zeros((12, n_pred))

for i in range(n_pred):
    lead_time = lead_time_arr[i]

    X, y, timey, y_persistance = pipeline(lead_time, return_persistance=True)

    pred_mean_full = np.array([])
    pred_std_full = np.array([])

    for decade in decades:
        K.clear_session()
        print(f'Predict: {1902+decade}-01-01 till {1911+decade}-12-01')

        ens_dir=f'ensemble_decade{decade}_lead{lead_time}'
        model = DEM()
        model.load(location=modeldir, dir_name=ens_dir)

        test_indeces = (timey>=f'{1902+decade}-01-01') & (timey<=f'{1911+decade}-12-01')
        testX, testy = X[test_indeces,:], y[test_indeces]
        pred_mean, pred_std = model.predict(testX)

        pred_mean_full = np.append(pred_mean_full, pred_mean)
        pred_std_full = np.append(pred_std_full, pred_std)

    #%% ===========================================================================
    # Deep ensemble
    # =============================================================================

    # all seasons skills
    all_season_corr[i] = np.corrcoef(y, pred_mean_full)[0,1]
    all_season_corr_pres[i] = np.corrcoef(y, y_persistance)[0,1]

    all_season_rmse[i] = rmse(y, pred_mean_full)
    all_season_rmse_pres[i] = rmse(y, y_persistance)

    all_season_nll[i] = model.evaluate(y, pred_mean_full, pred_std_full)

    # seasonal skills
    seas_corr[:, i] = correlation(y, pred_mean_full, timey - pd.tseries.offsets.MonthBegin(2))
    seas_corr_pers[:, i] = correlation(y, y_persistance, timey - pd.tseries.offsets.MonthBegin(2))
    seas_rmse[:, i] = rmse_mon(y, pred_mean_full, timey - pd.tseries.offsets.MonthBegin(2))

#%%
plt.close("all")

ax = plt.figure().gca()
plt.plot(lead_time_arr, all_season_corr, label="Deep Ensemble Mean")
plt.plot(lead_time_arr, all_season_corr_pres, label="Persistence")
plt.ylim(-0.2,1)
plt.xlim(0,8)
plt.xlabel('lead time')
plt.ylabel('r')
plt.title('Correlation skill')
plt.grid()
plt.legend()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax = plt.figure().gca()
plt.plot(lead_time_arr, all_season_rmse, label="Deep Ensemble Mean")
plt.plot(lead_time_arr, all_season_rmse_pres, label="Persistence")
plt.ylim(0.,1.8)
plt.xlim(0,8)
plt.xlabel('lead time')
plt.ylabel('RMSE')
plt.title('RMSE')
plt.grid()
plt.legend()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax = plt.figure().gca()
plt.plot(lead_time_arr, all_season_nll, label="Deep Ensemble")
plt.ylim(-0.5,0.5)
plt.xlim(0.,8)
plt.xlabel('lead time')
plt.ylabel('NLL')
plt.title('Negative-loglikelihood')
plt.grid()
plt.legend()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plot_seasonal_skill(lead_time_arr, seas_corr.T,  vmin=0, vmax=1)
plot_seasonal_skill(lead_time_arr, seas_corr_pers.T,  vmin=0, vmax=1)
plot_seasonal_skill(lead_time_arr, seas_rmse.T, vmin=0, vmax=1, cmap=plt.cm.Reds)