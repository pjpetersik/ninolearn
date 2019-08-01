# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.ticker import MaxNLocator

import pandas as pd

import keras.backend as K

from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr

from ninolearn.IO.read_post import data_reader
from ninolearn.learn.models.encoderDecoder import EncoderDecoder
from ninolearn.pathes import ed_model_dir
from ninolearn.private import plotdir
from ninolearn.utils import print_header
from ninolearn.learn.evaluation import rmse_monmean, correlation, rmse_mon
from ninolearn.plot.evaluation import plot_seasonal_skill

from os.path import join
plt.close("all")

#%%free memory from old sessions
K.clear_session()
decades_arr = [60, 70, 80, 90, 100, 110]
n_decades = len(decades_arr)

lead_time_arr = np.array([0, 3, 6, 9, 12, 15])
n_lead = len(lead_time_arr)

# scores for the full timeseries
all_season_corr = np.zeros(n_lead)
all_season_p = np.zeros(n_lead)
all_season_rmse = np.zeros(n_lead)

all_season_corr_pres = np.zeros(n_lead)
all_season_p_pers = np.zeros(n_lead)
all_season_rmse_pres = np.zeros(n_lead)

# decadal scores
decadel_corr = np.zeros((n_decades, n_lead))
decadel_p = np.zeros((n_decades, n_lead))
decadel_rmse = np.zeros((n_decades, n_lead))

decadel_corr_pres = np.zeros((n_decades, n_lead))
decadel_p_pers = np.zeros((n_decades, n_lead))
decadel_rmse_pres = np.zeros((n_decades, n_lead))

# scores for seasonal values
seas_corr = np.zeros((12, n_lead))
seas_p = np.zeros((12, n_lead))

seas_corr_pers = np.zeros((12, n_lead))
seas_p_pers = np.zeros((12, n_lead))

seas_rmse = np.zeros((12, n_lead))
seas_rmse_pers = np.zeros((12, n_lead))


# =============================================================================
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

# shift
shift = 3
for i in range(n_lead):
    lead = lead_time_arr[i]
    print_header(f'Lead time: {lead} months')

    y = yall[lead+shift:]
    X = Xall[:-lead-shift]

    timey = oni.index[lead+shift:]
    y_oni = oni[lead+shift:]
    y_oni_persistance = oni[:-lead-shift]
    y_oni_persistance.index = y_oni.index

    pred_full_oni = np.array([])
    pred_persistance_full_oni = np.array([])
    true_oni = np.array([])
    timeytrue = pd.DatetimeIndex([])

    pred_da_full = xr.zeros_like(label[lead+shift:,:,:])

    for j in range(n_decades):
        decade = decades_arr[j]

        print_header(f'Predict: {1902+decade}-01-01 till {1911+decade}-12-01')

        # get test data
        test_indeces = test_indeces = (timey>=f'{1902+decade}-01-01') & (timey<=f'{1911+decade}-12-01')
        testX, testy, testtimey = X[test_indeces,:], y[test_indeces,:], timey[test_indeces]
        testy_oni  = y_oni[test_indeces]

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
        pred_pers_oni = y_oni_persistance[test_indeces]

        # calculate the decadel scores of the ONI
        decadel_corr[j, i], decadel_p[j, i] = pearsonr(testy_oni, pred_oni)
        decadel_corr_pres[j, i], decadel_p_pers[j, i] = pearsonr(testy_oni, pred_pers_oni)

        decadel_rmse[j, i] = rmse_monmean(testy_oni, pred_oni, testtimey - pd.tseries.offsets.MonthBegin(1))
        decadel_rmse_pres[j, i] = rmse_monmean(testy_oni, pred_pers_oni, testtimey - pd.tseries.offsets.MonthBegin(1))

        # make the full time series
        pred_full_oni = np.append(pred_full_oni, pred_oni)
        true_oni = np.append(true_oni, testy_oni)
        pred_persistance_full_oni = np.append(pred_persistance_full_oni, pred_pers_oni)

    # correlation map
    pred_da_full_reshaped = pred_da_full.values.reshape(pred_da_full.shape[0],-1)
    label_reshaped = label[lead+shift:,:,:].values.reshape(pred_da_full.shape[0],-1)
    corr_map = np.zeros(label_reshaped.shape[1])

    for j in range(len(corr_map)):
        corr_map[j] = np.corrcoef(pred_da_full_reshaped[:,j], label_reshaped[:,j])[0,1]
    corr_map = corr_map.reshape((label.shape[1:]))

    # calculate all seasons scores ONI
    all_season_corr[i], all_season_p[i] = pearsonr(true_oni, pred_full_oni)
    all_season_corr_pres[i], all_season_p_pers[i] = pearsonr(true_oni, pred_persistance_full_oni)

    all_season_rmse[i] = rmse_monmean(true_oni, pred_full_oni, timeytrue - pd.tseries.offsets.MonthBegin(1))
    all_season_rmse_pres[i] = rmse_monmean(true_oni, pred_persistance_full_oni, timeytrue - pd.tseries.offsets.MonthBegin(1))


     # seasonal skills
    seas_corr[:, i], seas_p[:, i] = correlation(true_oni, pred_full_oni, timeytrue - pd.tseries.offsets.MonthBegin(1))
    seas_corr_pers[:, i], seas_p_pers[:, i] = correlation(true_oni, pred_persistance_full_oni, timeytrue - pd.tseries.offsets.MonthBegin(1))

    seas_rmse[:, i] = rmse_mon(true_oni, pred_full_oni, timeytrue - pd.tseries.offsets.MonthBegin(1))
    seas_rmse_pers[:, i] = rmse_mon(true_oni, pred_persistance_full_oni, timeytrue - pd.tseries.offsets.MonthBegin(1))


    # Plot correlation map
    fig, ax = plt.subplots(figsize=(8,2))
    plt.title(f"Lead time: {lead} month")
    C=ax.imshow(corr_map, origin='lower', vmin=0, vmax=1)
    plt.colorbar(C)
    plt.savefig(join(plotdir, f'ed_corr_map_lead{lead}.pdf'))


    # Plot ONI Forecasts
    plt.subplots(figsize=(8,1.8))

    plt.plot(timeytrue, pred_full_oni, c='navy')
    plt.plot(timey, y_oni, "k")

    plt.xlabel('Time [Year]')
    plt.ylabel('ONI [K]')

    plt.axhspan(-0.5, -6, facecolor='blue',  alpha=0.1,zorder=0)
    plt.axhspan(0.5, 6, facecolor='red', alpha=0.1,zorder=0)

    plt.xlim(timeytrue[0], timeytrue[-1])
    plt.ylim(-3,3)

    plt.title(f"Lead time: {lead} month")
    plt.grid()
    plt.tight_layout()
    plt.savefig(join(plotdir, f'ed_pred_lead{lead}.pdf'))


#%% =============================================================================
# Plot
# =============================================================================

decade_color = ['orange', 'violet', 'limegreen', 'darkgoldenrod', 'red', 'royalblue']
decade_name = ['1962-1971', '1972-1981', '1982-1991', '1992-2001', '2002-2011', '2012-2017']

# all season correlation score ONI
ax = plt.figure(figsize=(6.5,3.)).gca()
for j in range(n_decades):
    plt.plot(lead_time_arr, decadel_corr[j], c=decade_color[j], label=f"DE Mean ({decade_name[j]})")
    plt.plot(lead_time_arr, decadel_corr_pres[j], c=decade_color[j], linestyle='--', label=f"Persistence ({decade_name[j]})")
plt.plot(lead_time_arr, all_season_corr, 'k', label="DE Mean (1982-2017)", lw=2)
plt.plot(lead_time_arr, all_season_corr_pres,  'k', linestyle='--', label="Persistence (1982-2017)", lw=2)

plt.ylim(-0.2,1)
plt.xlim(0,lead_time_arr[-1])
plt.xlabel('Lead Time [Month]')
plt.ylabel('Correlation coefficient')
#plt.title('Correlation skill')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(join(plotdir, f'ed_all_season_corr.pdf'))

#%% all season rmse score
ax = plt.figure(figsize=(6.5,3.)).gca()

for j in range(n_decades):
    plt.plot(lead_time_arr, decadel_rmse[j], c=decade_color[j], label=f"DE Mean ({decade_name[j]})")
    plt.plot(lead_time_arr, decadel_rmse_pres[j], c=decade_color[j], linestyle='--', label=f"Persistence ({decade_name[j]})")
plt.plot(lead_time_arr, all_season_rmse, label="DE Mean (1982-2017)", c='k', lw=2)
plt.plot(lead_time_arr, all_season_rmse_pres, label="Persistence (1982-2017)", c='k', linestyle='--',  lw=2)

plt.ylim(0.,2)
plt.xlim(0, lead_time_arr[-1])
plt.xlabel('Lead Time [Month]')
plt.ylabel('SSRMSE')
#plt.title('Normalized RMSE')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(join(plotdir, f'ed_all_season_rmse.pdf'))


#%% contour skill plots ONI
plot_seasonal_skill(lead_time_arr, seas_corr.T,  vmin=0, vmax=1)
plt.contour(np.arange(1,13),lead_time_arr, seas_p.T, levels=[0.01, 0.05, 0.1], linestyles=['solid', 'dashed', 'dotted'], colors='k')
plt.title('Correlation skill')
plt.tight_layout()
plt.savefig(join(plotdir, f'ed_seasonal_corr.pdf'))

plot_seasonal_skill(lead_time_arr, seas_rmse.T, vmin=0, vmax=1.2, cmap=plt.cm.inferno_r, extend='max')
plt.title('SRMSE')
plt.tight_layout()
plt.savefig(join(plotdir, f'ed_seasonal_rmse.pdf'))
