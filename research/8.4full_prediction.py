# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras import backend as K

from sklearn.preprocessing import StandardScaler

from ninolearn.IO.read_post import data_reader

from ninolearn.learn.mlp import include_time_lag
from ninolearn.learn.dem import DEM
from ninolearn.pathes import modeldir


#%% =============================================================================
# read data
# =============================================================================
reader = data_reader(startdate='1981-01')

nino4 = reader.read_csv('nino4M')
nino34 = reader.read_csv('nino3.4M')
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

pca_u = reader.read_statistic('pca', variable='uwnd', dataset='NCEP', processed='anom')
pca2_u = pca_u['pca2']

#%% =============================================================================
#  process data
# =============================================================================
time_lag = 12
lead_time = 9
shift = 3 # actually 3

feature_unscaled = np.stack((nino34, nino12, nino3, nino4,
                             sc, yr,
                             wwv, iod,
                             pca2_u,
                             L_ssh, C_ssh, T_ssh, H_ssh, c2_ssh,
                             C_sst, H_sst,
                             S_air, T_air,
                             ), axis=1)

#feature_unscaled = np.stack((nino34,
#                             sc, yr,
#                             iod,
#                             L_ssh, C_ssh, T_ssh, H_ssh, c2_ssh,
#                             C_sst, H_sst,
#                             S_air, T_air,
#                             ), axis=1)

scaler = StandardScaler()
Xorg = scaler.fit_transform(feature_unscaled)

Xorg = np.nan_to_num(Xorg)

X = Xorg[:-lead_time-shift,:]
futureX = Xorg[-lead_time-shift-time_lag:,:]

X = include_time_lag(X, max_lag=time_lag)
futureX =  include_time_lag(futureX, max_lag=time_lag)

yorg = nino34.values
y = yorg[lead_time + time_lag + shift:]

timey = nino34.index[lead_time + time_lag + shift:]
futuretime = pd.date_range(start='2019-01-01',
                                        end=pd.to_datetime('2019-01-01')+pd.tseries.offsets.MonthEnd(lead_time+shift),
                                        freq='MS')

test_indeces = (timey>='2002-03-01') & (timey<='2011-02-01')
train_indeces = np.invert(test_indeces)

trainX, trainy, traintimey = X[train_indeces,:], y[train_indeces], timey[train_indeces]
testX, testy, testtimey = X[test_indeces,:], y[test_indeces], timey[test_indeces]

#%% =============================================================================
# Deep ensemble
# =============================================================================
plt.close("all")
index = -2

lead_time = np.array([0, 3, 6, 9])
n_pred = len(lead_time)

predfuture_mean = np.zeros(n_pred)
predfuture_std = np.zeros(n_pred)

predfuture_mean[0]  = nino34[index]
predtime = [timey[index]+pd.tseries.offsets.MonthBegin(1)]

for i in range(1, n_pred):
    K.clear_session()
    model = DEM()
    model.load(location=modeldir, dir_name=f'ensemble_lead{lead_time[i]}')
    predfuture_mean[i], predfuture_std[i] =  model.predict(futureX[index:index+1,:])

    predtime.append(timey[index]+pd.tseries.offsets.MonthBegin(lead_time[i]))

predtime = pd.DatetimeIndex(predtime)

# Prediction
plt.subplots(figsize=(7,3.5))
plt.axhspan(-0.5,
            -6,
            facecolor='blue',
            alpha=0.1,zorder=0)

plt.axhspan(0.5,
            6,
            facecolor='red',
            alpha=0.1,zorder=0)

plt.xlim(predtime[0],predtime[-1])
plt.ylim(-3,3)

std = 1.

# future
predictfuturey_p1std = predfuture_mean + np.abs(predfuture_std)
predictfuturey_m1std = predfuture_mean - np.abs(predfuture_std)
predictfuturey_p2std = predfuture_mean + 2 * np.abs(predfuture_std)
predictfuturey_m2std = predfuture_mean - 2 * np.abs(predfuture_std)

plt.fill_between(predtime, predictfuturey_m1std, predictfuturey_p1std , facecolor='orange', alpha=0.7)
plt.fill_between(predtime, predictfuturey_m2std, predictfuturey_p2std , facecolor='orange', alpha=0.3)
plt.plot(predtime, predfuture_mean, "darkorange")
plt.grid()