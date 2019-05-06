# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K

from sklearn.preprocessing import StandardScaler

from ninolearn.IO.read_post import data_reader

from ninolearn.learn.mlp import include_time_lag
from ninolearn.learn.dem import DEM
from ninolearn.pathes import modeldir
from ninolearn.learn.evaluation import rmse, correlation, rmse_mon
from ninolearn.plot.evaluation import plot_monthly_skill

from matplotlib.ticker import MaxNLocator
#%% =============================================================================
# read data
# =============================================================================
reader = data_reader(startdate='1981-01')

nino4 = reader.read_csv('nino4M')
nino34 = reader.read_csv('nino3.4S')
nino12 = reader.read_csv('nino1+2M')
nino3 = reader.read_csv('nino3M')

iod = reader.read_csv('iod')
wwv = reader.read_csv('wwv')

len_ts = len(nino34)
sc = np.cos(np.arange(len_ts)/12*2*np.pi)
yr = np.arange(len_ts) % 12

network_ssh = reader.read_statistic('network_metrics', variable='sshg',
                           dataset='GODAS', processed="anom")

network_sst = reader.read_statistic('network_metrics', variable='sst',
                           dataset='ERSSTv5', processed="anom")

network_sat = reader.read_statistic('network_metrics', variable='air',
                           dataset='NCEP', processed="anom")

c2_ssh = network_ssh['fraction_clusters_size_2']
S_ssh = network_ssh['fraction_giant_component']
H_ssh = network_ssh['corrected_hamming_distance']
T_ssh = network_ssh['global_transitivity']
C_ssh = network_ssh['avelocal_transmissivity']
L_ssh = network_ssh['average_path_length']

C_sst = network_sst['avelocal_transmissivity']
H_sst = network_ssh['corrected_hamming_distance']

S_air = network_ssh['fraction_giant_component']
T_air = network_ssh['global_transitivity']

#%% =============================================================================
#  process data
# =============================================================================
time_lag = 12
lead_time_arr = np.array([1, 2, 3, 4, 5, 6, 9, 12])
n_pred = len(lead_time_arr)
all_season_corr = np.zeros(n_pred)
all_season_rmse = np.zeros(n_pred)
all_season_corr_pres = np.zeros(n_pred)
all_season_rmse_pres = np.zeros(n_pred)
all_season_nll = np.zeros(n_pred)

monthly_corr = np.zeros((12, n_pred))
monthly_rmse = np.zeros((12, n_pred))


for i in range(n_pred):
    K.clear_session()
    lead_time = lead_time_arr[i]
    feature_unscaled = np.stack((nino34, sc, yr, iod, wwv,
                                 L_ssh, C_ssh, T_ssh, H_ssh, c2_ssh,
                                 C_sst, H_sst,
                                 S_air, T_air,
                                 ), axis=1)

    scaler = StandardScaler()
    Xorg = scaler.fit_transform(feature_unscaled)
    Xorg = np.nan_to_num(Xorg)

    X = Xorg[:-lead_time,:]
    X = include_time_lag(X, max_lag=time_lag)


    yorg = nino34.values
    y = yorg[time_lag +  lead_time:]

    ypres = yorg[time_lag:-lead_time]

    timey = nino34.index[lead_time + time_lag:]

    test_indeces = (timey>='2002-01-01') & (timey<='2011-03-01')
    train_indeces = np.invert(test_indeces)

    testX, testy, testtimey = X[test_indeces,:], y[test_indeces], timey[test_indeces]

    pred_pres = ypres[test_indeces]
    #%% ===========================================================================
    # Deep ensemble
    # =============================================================================

    model = DEM()
    print('load')
    model.load(location=modeldir, dir_name=f'ensemble_lead{lead_time}')
    print('predict')
    pred_mean, pred_std =  model.predict(testX)

    # all seasons skills
    all_season_corr[i] = np.corrcoef(testy, pred_mean)[0,1]
    all_season_corr_pres[i] = np.corrcoef(testy, pred_pres)[0,1]

    all_season_rmse[i] = rmse(testy, pred_mean)
    all_season_rmse_pres[i] = rmse(testy, pred_pres)

    all_season_nll[i] = model.evaluate(testy, pred_mean, pred_std)

    # monthly skills
    monthly_corr[:, i] = correlation(testy, pred_mean, testtimey)
    monthly_rmse[:, i] = rmse_mon(testy, pred_mean, testtimey)

nino_mean = np.mean(yorg)
nino_std = np.std(yorg)

nll_nino = model.evaluate(testy, nino_mean, nino_std)

#%%
plt.close("all")

ax = plt.figure().gca()
plt.plot(lead_time_arr - 1, all_season_corr, label="Deep Ensemble Mean")
plt.plot(lead_time_arr - 1, all_season_corr_pres, label="Persistence")
plt.ylim(-0.2,1)
plt.xlim(0,8)
plt.xlabel('lead time')
plt.ylabel('r')
plt.title('Correlation skill')
plt.grid()
plt.legend()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax = plt.figure().gca()
plt.plot(lead_time_arr - 1, all_season_rmse, label="Deep Ensemble Mean")
plt.plot(lead_time_arr - 1, all_season_rmse_pres, label="Persistence")
plt.ylim(0.,1)
plt.xlim(0,8)
plt.xlabel('lead time')
plt.ylabel('RMSE')
plt.title('RMSE')
plt.grid()
plt.legend()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))


ax = plt.figure().gca()
plt.plot(lead_time_arr - 1, all_season_nll, label="Deep Ensemble")
plt.hlines(nll_nino,0,12, label='Nino3.4 mean/std', color='orange')
plt.ylim(-0.5,0.5)
plt.xlim(0.,8)
plt.xlabel('lead time')
plt.ylabel('NLL')
plt.title('Negative-loglikelihood')
plt.grid()
plt.legend()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plot_monthly_skill(lead_time_arr-1, monthly_corr.T)
plot_monthly_skill(lead_time_arr-1, monthly_rmse.T, vmin=0, vmax=1)