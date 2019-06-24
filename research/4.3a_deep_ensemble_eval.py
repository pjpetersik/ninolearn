# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from os.path import join

from keras import backend as K

from ninolearn.learn.dem import DEM
from ninolearn.pathes import modeldir
from ninolearn.learn.evaluation import rmse_monmean, correlation, rmse_mon, rmse, seasonal_nll
from ninolearn.plot.prediction import plot_prediction
from ninolearn.plot.evaluation import plot_seasonal_skill
from ninolearn.utils import print_header
from data_pipeline import pipeline
from ninolearn.private import plotdir

from scipy.stats import pearsonr
plt.close("all")

#%% =============================================================================
#  process data
# =============================================================================
decades_arr = [80, 90, 100, 110]
n_decades = len(decades_arr)

lead_time_arr = np.array([0, 3, 6, 9, 12, 15, 18])
n_lead = len(lead_time_arr)

# scores for the full timeseries
all_season_corr = np.zeros(n_lead)
all_season_p = np.zeros(n_lead)
all_season_rmse = np.zeros(n_lead)

all_season_corr_pres = np.zeros(n_lead)
all_season_p_pers = np.zeros(n_lead)
all_season_rmse_pres = np.zeros(n_lead)

all_season_nll = np.zeros(n_lead)

# decadal scores
decadel_corr = np.zeros((n_decades, n_lead))
decadel_p = np.zeros((n_decades, n_lead))
decadel_rmse = np.zeros((n_decades, n_lead))

decadel_corr_pres = np.zeros((n_decades, n_lead))
decadel_p_pers = np.zeros((n_decades, n_lead))
decadel_rmse_pres = np.zeros((n_decades, n_lead))

decadel_nll = np.zeros((n_decades, n_lead))

# scores for seasonal values
seas_corr = np.zeros((12, n_lead))
seas_p = np.zeros((12, n_lead))

seas_corr_pers = np.zeros((12, n_lead))
seas_p_pers = np.zeros((12, n_lead))

seas_rmse = np.zeros((12, n_lead))
seas_rmse_pers = np.zeros((12, n_lead))

seas_nll = np.zeros((12, n_lead))

for i in range(n_lead):
    lead_time = lead_time_arr[i]
    print_header(f'Lead time: {lead_time} months')

    X, y, timey, y_persistance = pipeline(lead_time, return_persistance=True)

    pred_mean_full = np.array([])
    pred_std_full = np.array([])
    pred_persistance_full = np.array([])
    ytrue = np.array([])
    timeytrue = pd.DatetimeIndex([])

    for j in range(n_decades):
        decade = decades_arr[j]

        # free some memory
        K.clear_session()

        # make predictions
        print(f'Predict: {1902+decade}-01-01 till {1911+decade}-12-01')

        ens_dir=f'ensemble_decade{decade}_lead{lead_time}'
        model = DEM()
        model.load(location=modeldir, dir_name=ens_dir)

        if decade == 80:
            test_indeces = (timey>=f'{1903+decade}-01-01') & (timey<=f'{1911+decade}-12-01')
        else:
            test_indeces = (timey>=f'{1902+decade}-01-01') & (timey<=f'{1911+decade}-12-01')

        testX, testy, testtimey = X[test_indeces,:], y[test_indeces], timey[test_indeces]

        pred_mean, pred_std = model.predict(testX)
        pred_pers = y_persistance[test_indeces]

        # calculate the decadel scores
        decadel_corr[j, i], decadel_p[j, i] = pearsonr(testy, pred_mean)
        decadel_corr_pres[j, i], decadel_p_pers[j, i] = pearsonr(testy, pred_pers)

        decadel_rmse[j, i] = rmse_monmean(testy, pred_mean, testtimey - pd.tseries.offsets.MonthBegin(1))
        decadel_rmse_pres[j, i] = rmse_monmean(testy, pred_pers, testtimey - pd.tseries.offsets.MonthBegin(1))

        decadel_nll[j, i] = model.evaluate(testy, pred_mean, pred_std)

        # make the full time series
        pred_mean_full = np.append(pred_mean_full, pred_mean)
        pred_std_full = np.append(pred_std_full, pred_std)
        pred_persistance_full = np.append(pred_persistance_full, pred_pers)
        ytrue = np.append(ytrue, testy)
        timeytrue = timeytrue.append(testtimey)

    #%% ===========================================================================
    # Deep ensemble
    # =============================================================================

    # calculate all seasons scores
    all_season_corr[i], all_season_p[i] = pearsonr(ytrue, pred_mean_full)
    all_season_corr_pres[i], all_season_p_pers[i] = pearsonr(ytrue, pred_persistance_full)

    all_season_rmse[i] = rmse_monmean(ytrue, pred_mean_full, timeytrue - pd.tseries.offsets.MonthBegin(1))
    all_season_rmse_pres[i] = rmse_monmean(ytrue, pred_persistance_full, timeytrue - pd.tseries.offsets.MonthBegin(1))

    all_season_nll[i] = model.evaluate(ytrue, pred_mean_full, pred_std_full)

    # seasonal skills
    seas_corr[:, i], seas_p[:, i] = correlation(ytrue, pred_mean_full, timeytrue - pd.tseries.offsets.MonthBegin(1))
    seas_corr_pers[:, i], seas_p_pers[:, i] = correlation(ytrue, pred_persistance_full, timeytrue - pd.tseries.offsets.MonthBegin(1))

    seas_rmse[:, i] = rmse_mon(ytrue, pred_mean_full, timeytrue - pd.tseries.offsets.MonthBegin(1))
    seas_rmse_pers[:, i] = rmse_mon(ytrue, pred_persistance_full, timeytrue - pd.tseries.offsets.MonthBegin(1))

    seas_nll[:, i] = seasonal_nll(ytrue, pred_mean_full, pred_std_full,
                                  timeytrue - pd.tseries.offsets.MonthBegin(1), model.evaluate)

    plt.subplots(figsize=(8,1.8))
    # plot prediction
    plot_prediction(timeytrue, pred_mean_full, std=pred_std_full, facecolor='royalblue', line_color='navy')

    # observation
    plt.plot(timey, y, "k")
    plt.xlabel('Time [Year]')
    plt.ylabel('ONI [K]')

    plt.axhspan(-0.5, -6, facecolor='blue',  alpha=0.1,zorder=0)
    plt.axhspan(0.5, 6, facecolor='red', alpha=0.1,zorder=0)

    plt.xlim(timeytrue[0],timeytrue[-1])
    plt.ylim(-3,3)

    plt.title(f"Lead time: {lead_time} month")
    plt.grid()
    plt.tight_layout()
    plt.savefig(join(plotdir, f'pred_lead{lead_time}.pdf'))

#%%
decade_color = ['limegreen', 'darkgoldenrod', 'red', 'royalblue']
decade_name = ['1983-1991', '1992-2001', '2002-2011', '2012-2018']

# all season correlation score
ax = plt.figure(figsize=(6.5,3.)).gca()
for j in range(n_decades):
    plt.plot(lead_time_arr, decadel_corr[j], c=decade_color[j], label=f"DE Mean ({decade_name[j]})")
    plt.plot(lead_time_arr, decadel_corr_pres[j], c=decade_color[j], linestyle='--', label=f"Persistence ({decade_name[j]})")
plt.plot(lead_time_arr, all_season_corr, 'k', label="DE Mean (1982-2018)", lw=2)
plt.plot(lead_time_arr, all_season_corr_pres,  'k', linestyle='--', label="Persistence (1982-2018)", lw=2)

plt.ylim(-0.2,1)
plt.xlim(0,18)
plt.xlabel('Lead Time [Month]')
plt.ylabel('Correlation coefficient')
#plt.title('Correlation skill')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(join(plotdir, f'all_season_corr.pdf'))

#%% all season rmse score
ax = plt.figure(figsize=(6.5,3.)).gca()

for j in range(n_decades):
    plt.plot(lead_time_arr, decadel_rmse[j], c=decade_color[j], label=f"DE Mean ({decade_name[j]})")
    plt.plot(lead_time_arr, decadel_rmse_pres[j], c=decade_color[j], linestyle='--', label=f"Persistence ({decade_name[j]})")
plt.plot(lead_time_arr, all_season_rmse, label="DE Mean (1982-2018)", c='k', lw=2)
plt.plot(lead_time_arr, all_season_rmse_pres, label="Persistence (1982-2018)", c='k', linestyle='--',  lw=2)

plt.ylim(0.,2)
plt.xlim(0,18)
plt.xlabel('Lead Time [Month]')
plt.ylabel('SSRMSE')
#plt.title('Normalized RMSE')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(join(plotdir, f'all_season_rmse.pdf'))

#%% all seasons negative loglikelihood
ax = plt.figure(figsize=(6.5,3.)).gca()
for j in range(n_decades):
    plt.plot(lead_time_arr, decadel_nll[j], c=decade_color[j], label=f"Deep Ens.  ({decade_name[j]})")
plt.plot(lead_time_arr, all_season_nll, label="Deep Ens.  (1983-2018)", c='k', lw=2)

plt.ylim(-0.5,0.7)
plt.xlim(0.,18)
plt.xlabel('Lead Time [Month]')
plt.ylabel('NLL')
#plt.title('Negative-loglikelihood')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(join(plotdir, f'all_season_nll.pdf'))

#%% contour skill plots
plot_seasonal_skill(lead_time_arr, seas_corr.T,  vmin=0, vmax=1)
plt.contour(np.arange(1,13),lead_time_arr, seas_p.T, levels=[0.01, 0.05, 0.1], linestyles=['solid', 'dashed', 'dotted'], colors='k')
plt.title('Correlation skill')
plt.tight_layout()
plt.savefig(join(plotdir, f'seasonal_corr.pdf'))

plot_seasonal_skill(lead_time_arr, seas_rmse.T, vmin=0, vmax=1.2, cmap=plt.cm.inferno_r, extend='max')
plt.title('SRMSE')
plt.tight_layout()
plt.savefig(join(plotdir, f'seasonal_rmse.pdf'))

plot_seasonal_skill(lead_time_arr, seas_nll.T, vmin=-1, vmax=1, cmap=plt.cm.inferno_r, extend='both')
plt.title('NLL')
plt.tight_layout()
plt.savefig(join(plotdir, f'seasonal_nll.pdf'))


#%% FOR ENSO ML Paper
#plt.close("all")
#plot_seasonal_skill(lead_time_arr, seas_corr_pers.T,  vmin=-1, vmax=1, cmap=plt.cm.seismic, extend='neither')
#
#seas_p_pers_pos = seas_p_pers.copy()
#seas_p_pers_pos[seas_corr_pers<0] = 1
#
#plt.contour(np.arange(1,13),lead_time_arr, seas_p_pers_pos.T, levels=[0.01, 0.05, 0.1])
#plot_seasonal_skill(lead_time_arr, seas_p_pers_pos.T,  vmin=0, vmax=1., cmap=plt.cm.Blues, extend='max')