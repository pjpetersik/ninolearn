# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import keras.backend as K

from sklearn.preprocessing import StandardScaler

from ninolearn.IO.read_post import data_reader
from ninolearn.plot.evaluation  import plot_correlation
from ninolearn.plot.prediction import plot_prediction
from ninolearn.learn.evaluation import rmse
from ninolearn.learn.mlp import include_time_lag
from ninolearn.learn.dem import DEM
from ninolearn.utils import print_header
from ninolearn.pathes import modeldir

#%%
K.clear_session()

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

pca_u = reader.read_statistic('pca', variable='uwnd', dataset='NCEP', processed='anom')
pca2_u = pca_u['pca2']

#%% =============================================================================
#  process data
# =============================================================================
time_lag = 12
lead_time = 3
shift = 3 # actually 3

feature_unscaled = np.stack((nino34, #nino12, nino3, nino4,
                             sc, #yr,
                             wwv, iod,
                             pca2_u,
                             c2_ssh,
#                             L_ssh, C_ssh, T_ssh, H_ssh, c2_ssh,
#                             C_sst, H_sst,
#                             S_air, T_air,
                             ), axis=1)

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

#%% =============================================================================
# Deep ensemble
# =============================================================================
decades = [80, 90, 100, 110]

pred_mean_full = np.array([])
pred_std_full = np.array([])

decadal_corr = np.zeros(len(decades))
decadal_rmse = np.zeros(len(decades))
decadal_nll = np.zeros(len(decades))

i=0
for decade in decades:
    print_header(f'{1902+decade}-01-01 till {1911+decade}-12-01')

    test_indeces = (timey>=f'{1902+decade}-01-01') & (timey<=f'{1911+decade}-12-01')
    train_indeces = np.invert(test_indeces)

    trainX, trainy = X[train_indeces,:], y[train_indeces]
    testX, testy = X[test_indeces,:], y[test_indeces]


    model = DEM()

    model.set_parameters(layers=1, dropout=0.2, noise=0.2, l1_hidden=0.0,
                l2_hidden=0.2, l1_mu=0., l2_mu=0.2, l1_sigma=0.0, l2_sigma=0.2,
                lr=0.001, batch_size=1, epochs=500, n_segments=5, n_members_segment=1, patience=30, verbose=0, std=True)

    model.fit(trainX, trainy)

    pred_mean, pred_std = model.predict(testX)

    score = model.evaluate(testy, pred_mean, pred_std)
    print_header(f"Score: {score}")

    pred_mean_full = np.append(pred_mean_full, pred_mean)
    pred_std_full = np.append(pred_std_full, pred_std)

    if model.std:
        ens_dir=f'ensemble_decade{decade}_lead{lead_time}'
    else:
        ens_dir=f'ensemble_decade{decade}_lead{lead_time}'

    model.save(location=modeldir, dir_name=ens_dir)

    decadal_nll[i] = score
    decadal_rmse[i] = round(rmse(testy, pred_mean),2)
    decadal_corr[i] = np.corrcoef(testy, pred_mean)[0,1]

    del model
    i+=1
##%% just for testing the loading function delete and load the model
#del model
#model = DEM()
#model.load(location=modeldir, dir_name=ens_dir)
#
#pred_mean, pred_std = model.predict(testX)
#predtrain_mean, predtrain_std = model.predict(trainX)
#predfuture_mean, predfuture_std =  model.predict(futureX)

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

plt.xlim(timey[0],futuretime[-1])
plt.ylim(-3,3)

# test
plot_prediction(timey, pred_mean_full, std=pred_std_full, facecolor='royalblue', line_color='navy')

# observation
plt.plot(timey, y, "k")

pred_rmse = round(rmse(y, pred_mean_full),2)
plt.title(f"Lead time: {lead_time} month, RMSE (of mean): {pred_rmse}")
plt.grid()

# plot explained variance
# minus two month to center season around central month
plot_correlation(y, pred_mean_full, timey - pd.tseries.offsets.MonthBegin(2))


# Error distribution
plt.subplots()
plt.title("Error distribution")
error = pred_mean_full - y

plt.hist(error, bins=16)

#decadal scores
decades_plot = ['80s', '90s', '00s', '10s']

fig, ax = plt.subplots(1,3)
ax[0].set_title("NLL")
ax[0].bar(decades_plot, decadal_nll)

ax[1].set_title("correlation")
ax[1].bar(decades_plot, decadal_corr)

ax[2].set_title("rmse")
ax[2].bar(decades_plot, decadal_rmse)


