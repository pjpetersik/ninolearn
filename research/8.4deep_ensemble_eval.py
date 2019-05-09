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
from ninolearn.utils import print_header
from data_pipeline import pipeline

#%% =============================================================================
#  process data
# =============================================================================
decades = [80, 90, 100, 110]
#decades = [100]

lead_time_arr = np.array([0, 3, 6, 9, 12, 15])

n_pred = len(lead_time_arr)
all_season_corr = np.zeros(n_pred)
all_season_rmse = np.zeros(n_pred)
all_season_corr_pres = np.zeros(n_pred)
all_season_rmse_pres = np.zeros(n_pred)
all_season_nll = np.zeros(n_pred)

seas_corr = np.zeros((12, n_pred))
seas_corr_pers = np.zeros((12, n_pred))
seas_rmse = np.zeros((12, n_pred))
seas_rmse_pers = np.zeros((12, n_pred))

for i in range(n_pred):
    lead_time = lead_time_arr[i]
    print_header(f'Lead time: {lead_time} months')

    X, y, timey, y_persistance = pipeline(lead_time, return_persistance=True)

    pred_mean_full = np.array([])
    pred_std_full = np.array([])
    pred_persistance_full = np.array([])
    ytrue = np.array([])
    timeytrue = pd.DatetimeIndex([])

    for decade in decades:
        K.clear_session()
        print(f'Predict: {1902+decade}-01-01 till {1911+decade}-12-01')

        ens_dir=f'ensemble_decade{decade}_lead{lead_time}'
        model = DEM()
        model.load(location=modeldir, dir_name=ens_dir)

        test_indeces = (timey>=f'{1902+decade}-01-01') & (timey<=f'{1911+decade}-12-01')
        testX, testy, testtimey = X[test_indeces,:], y[test_indeces], timey[test_indeces]

        pred_mean, pred_std = model.predict(testX)
        pred_pers = y_persistance[test_indeces]

        pred_mean_full = np.append(pred_mean_full, pred_mean)
        pred_std_full = np.append(pred_std_full, pred_std)
        pred_persistance_full = np.append(pred_persistance_full, pred_pers)
        ytrue = np.append(ytrue, testy)
        timeytrue= timeytrue.append(testtimey)

    #%% ===========================================================================
    # Deep ensemble
    # =============================================================================

    # all seasons skills
    all_season_corr[i] = np.corrcoef(ytrue, pred_mean_full)[0,1]
    all_season_corr_pres[i] = np.corrcoef(ytrue, pred_persistance_full)[0,1]

    all_season_rmse[i] = rmse(ytrue, pred_mean_full)
    all_season_rmse_pres[i] = rmse(ytrue, pred_persistance_full)

    all_season_nll[i] = model.evaluate(ytrue, pred_mean_full, pred_std_full)

    # seasonal skills
    seas_corr[:, i] = correlation(ytrue, pred_mean_full, timeytrue - pd.tseries.offsets.MonthBegin(1))
    seas_corr_pers[:, i] = correlation(ytrue, pred_persistance_full, timeytrue - pd.tseries.offsets.MonthBegin(1))

    seas_rmse[:, i] = rmse_mon(ytrue, pred_mean_full, timeytrue - pd.tseries.offsets.MonthBegin(1))
    seas_rmse_pers[:, i] = rmse_mon(ytrue, pred_persistance_full, timeytrue - pd.tseries.offsets.MonthBegin(1))

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
plt.vlines(4,0,9)
plt.plot(np.arange(7,13), np.arange(0,6),'k')
plt.plot(np.arange(1,5), np.arange(6,10),'k')

plot_seasonal_skill(lead_time_arr, seas_corr_pers.T,  vmin=0, vmax=1)
plt.vlines(4,0,9)
plt.plot(np.arange(7,13), np.arange(0,6),'k')
plt.plot(np.arange(1,5), np.arange(6,10),'k')
plot_seasonal_skill(lead_time_arr, seas_corr.T-seas_corr_pers.T,  vmin=-1, vmax=1, extend='both', cmap=plt.cm.bwr)

plot_seasonal_skill(lead_time_arr, seas_rmse.T, vmin=0, vmax=1, cmap=plt.cm.Reds)
plot_seasonal_skill(lead_time_arr, seas_rmse_pers.T, vmin=0, vmax=1, cmap=plt.cm.Reds)