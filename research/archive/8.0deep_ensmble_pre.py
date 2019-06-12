"""
In this script, I want to build a deep ensemble that is first trained on the GFDL data
and then trained on the observations
"""
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

K.clear_session()

def mixture(pred):
    """
    returns the ensemble mixture results
    """
    mix_mean = pred[:,0,:].mean(axis=1)
    mix_var = np.mean(pred[:,0,:]**2 + pred[:,1,:]**2, axis=1)  - mix_mean**2
    mix_std = np.sqrt(mix_var)
    return mix_mean, mix_std

#%% =============================================================================
# #%% read data
# =============================================================================
reader = data_reader(startdate='1701-01', enddate='2199-12')

nino34 = reader.read_csv('nino3.4M_gfdl')
iod = reader.read_csv('iod_gfdl')

# PCA data
pca_air = reader.read_statistic('pca', variable='tas',
                           dataset='GFDL-CM3', processed="anom")

pca2_air = pca_air['pca2']

# Network metics
nwm_ssh = reader.read_statistic('network_metrics', 'zos',
                                        dataset='GFDL-CM3',
                                        processed='anom')

c2_ssh = nwm_ssh['fraction_clusters_size_2']
S_ssh = nwm_ssh['fraction_giant_component']
H_ssh = nwm_ssh['corrected_hamming_distance']
T_ssh = nwm_ssh['global_transitivity']
C_ssh = nwm_ssh['avelocal_transmissivity']
L_ssh = nwm_ssh['average_path_length']

nwm_air = reader.read_statistic('network_metrics', 'tas',
                                        dataset='GFDL-CM3',
                                        processed='anom')

S_air = nwm_air['fraction_giant_component']
T_air = nwm_air['global_transitivity']

nwm_sst = reader.read_statistic('network_metrics', 'tos',
                                        dataset='GFDL-CM3',
                                        processed='anom')

H_sst = nwm_sst['corrected_hamming_distance']
C_sst = nwm_sst['avelocal_transmissivity']

# artificiial data
len_ts = len(nino34)
sc = np.cos(np.arange(len_ts)/12*2*np.pi)
yr =  np.arange(len_ts) % 12

#%% ===========================================================================
# # process data
# =============================================================================
time_lag = 12
lead_time = 3
shift = 2 # actually 3

feature_unscaled = np.stack((nino34,
                             sc, yr,
                             iod,
                             L_ssh, C_ssh, T_ssh, H_ssh, c2_ssh,
                             C_sst, H_sst,
                             S_air, T_air,
                             ), axis=1)

scaler = StandardScaler()
Xorg = scaler.fit_transform(feature_unscaled)

Xorg = np.nan_to_num(Xorg)

X = Xorg[:-lead_time-shift,:]

X = include_time_lag(X, max_lag=time_lag)

yorg = nino34.values
y = yorg[lead_time + time_lag + shift:]

timey = nino34.index[lead_time + time_lag + shift:]

test_indeces = (timey>='2100-03-01') & (timey<='2199-12-01')
train_indeces = np.invert(test_indeces)

trainX, trainy, traintimey = X[train_indeces,:], y[train_indeces], timey[train_indeces]
testX, testy, testtimey = X[test_indeces,:], y[test_indeces], timey[test_indeces]

model = DEM()

model.set_parameters(layers=1, dropout=0.05, noise=0.6, l1_hidden=0.1,
            l2_hidden=0.02, l1_mu=0.1, l2_mu=0.1, l1_sigma=0.0, l2_sigma=0.0,
            lr=0.01, n_segments=1, n_members_segment=1, patience=2, epochs=100, verbose=1, std=True)

model.fit(trainX, trainy, testX, testy)
#%%
pred_mean, pred_std = model.predict(testX)

score = model.evaluate(testy, pred_mean, pred_std)
print_header(f"Score: {score}")

if model.std:
    ens_dir=f'pre_ensemble_lead{lead_time}'
else:
    ens_dir=f'pre_simple_ensemble_lead{lead_time}'

model.save_weights(location=modeldir, dir_name=ens_dir)
