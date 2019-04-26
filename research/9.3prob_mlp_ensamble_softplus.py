# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Input, concatenate
from keras.layers import LSTM
from keras.layers import Dropout, GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.layers.core import Lambda

from sklearn.preprocessing import MinMaxScaler,StandardScaler

from ninolearn.IO.read_post import data_reader
from ninolearn.plot.evaluation  import plot_explained_variance
from ninolearn.learn.evaluation import nrmse, rmse, inside_fraction
from ninolearn.learn.mlp import include_time_lag
from ninolearn.learn.losses import nll_gaussian
from ninolearn.learn.augment import window_warping
from ninolearn.utils import print_header

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
reader = data_reader(startdate='1981-01')

nino4 = reader.read_csv('nino4M')
nino34 = reader.read_csv('nino3.4M')
nino12 = reader.read_csv('nino1+2M')
nino3 = reader.read_csv('nino3M')

iod = reader.read_csv('iod')
wwv = reader.read_csv('wwv')

len_ts = len(nino34)
sc = np.cos(np.arange(len_ts)/12*2*np.pi)
yr =  np.arange(len_ts) % 12

network_ssh = reader.read_statistic('network_metrics', variable='sshg',
                           dataset='GODAS', processed="anom")

network_sst = reader.read_statistic('network_metrics', variable='sshg',
                           dataset='GODAS', processed="anom")

c2_ssh = network_ssh['fraction_clusters_size_2']
S_ssh = network_ssh['fraction_giant_component']
H_ssh = network_ssh['corrected_hamming_distance']
T_ssh = network_ssh['global_transitivity']
C_ssh = network_ssh['avelocal_transmissivity']
L_ssh = network_ssh['average_path_length']

C_sst = network_sst['avelocal_transmissivity']
H_sst = network_ssh['corrected_hamming_distance']


pca_air = reader.read_statistic('pca', variable='air',
                           dataset='NCEP', processed="anom")
pca_u = reader.read_statistic('pca', variable='uwnd',
                           dataset='NCEP', processed="anom")
pca_v = reader.read_statistic('pca', variable='vwnd',
                           dataset='NCEP', processed="anom")
pca_sst = reader.read_statistic('pca', variable='sst',
                           dataset='ERSSTv5', processed="anom")

pca2_air = pca_air['pca2']
pca2_u = pca_u['pca2']
pca2_v = pca_v['pca2']
pca2_sst = pca_sst['pca2']

#%% =============================================================================
# # process data
# =============================================================================
time_lag = 6
lead_time = 6

feature_unscaled = np.stack((nino34,
                             sc,
                             wwv, iod,
                             L_ssh, C_ssh, T_ssh, H_ssh, c2_ssh,
                             C_sst, H_sst,
                             pca2_u, pca2_v, pca2_air, pca2_sst
                             ), axis=1)

scaler = StandardScaler()
Xorg = scaler.fit_transform(feature_unscaled)

Xorg = np.nan_to_num(Xorg)

X = Xorg[:-lead_time,:]
futureX = Xorg[-lead_time-time_lag:,:]

X = include_time_lag(X, max_lag=time_lag)
futureX =  include_time_lag(futureX, max_lag=time_lag)

yorg = nino34.values
y = yorg[lead_time + time_lag:]

timey = nino34.index[lead_time + time_lag:]
futuretime = pd.date_range(start='2019-01-01',
                                        end=pd.to_datetime('2019-01-01')+pd.tseries.offsets.MonthEnd(lead_time),
                                        freq='MS')

test_indeces = (timey>='2002-01-01') & (timey<='2018-12-01')
train_indeces = np.invert(test_indeces)

trainX, trainy, traintimey = X[train_indeces,:], y[train_indeces], timey[train_indeces]
testX, testy, testtimey = X[test_indeces,:], y[test_indeces], timey[test_indeces]

#%% =============================================================================
# # neural network
# =============================================================================

optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

es = EarlyStopping(monitor='val_loss',
                              min_delta=0.0,
                              patience=10,
                              verbose=0,
                              mode='min',
                              restore_best_weights=True)

model_ens = []
n_ens = 5
segment_len = trainX.shape[0]//n_ens
n_ens_sel=0

l1 = 0.01
l2 = 0.1

l1_out = 0.0
l2_out = 0.1

while n_ens_sel<n_ens:

    # define the model
    inputs = Input(shape=(trainX.shape[1],))
    h = GaussianNoise(0.1)(inputs)

    h = Dense(8, activation='relu',
              kernel_regularizer=regularizers.l1_l2(l1, l2))(h)
    h = Dropout(0.2)(h)
    mu = Dense(1, activation='linear', kernel_regularizer=regularizers.l1_l2(l1_out, l2_out))(h)
    sigma = Dense(1, activation='softplus', kernel_regularizer=regularizers.l1_l2(l1_out, l2_out))(h)

    outputs = concatenate([mu, sigma])

    model_ens.append(Model(inputs=inputs, outputs=outputs))
    model_ens[-1].compile(loss=nll_gaussian, optimizer=optimizer)

    print_header(f"Train Iteration Nr {n_ens_sel}")
    start_ind = n_ens_sel * segment_len
    end_ind = (n_ens_sel+1) * segment_len

    history = model_ens[-1].fit(np.delete(trainX, np.s_[start_ind:end_ind], axis=0),
                                np.delete(trainy, np.s_[start_ind:end_ind]),
                                epochs=300, batch_size=10, verbose=1,
                                shuffle=True, callbacks=[es],
                                validation_data=(trainX[start_ind:end_ind], trainy[start_ind:end_ind]))

    mem_pred = model_ens[-1].predict(trainX)
    mem_mean = mem_pred[:,0]
    mem_std = np.abs(mem_pred[:,1])

    in_frac = inside_fraction(trainy, mem_mean, mem_std)

    if in_frac > 0.8 or in_frac < 0.55:
        print_header("Reject this model. Unreasonable stds.")
        model_ens.pop()

    elif rmse(trainy, mem_mean)>1.:
        print_header(f"Reject this model. Unreasonalble rmse of {rmse(trainy, mem_mean)}")
        model_ens.pop()

    elif np.min(history.history["loss"])>1:
        print_header("Reject this model. High minimum loss.")
        model_ens.pop()

    n_ens_sel = len(model_ens)

#%%


pred_ens = np.zeros((len(testy),2, n_ens_sel))
predtrain_ens = np.zeros((len(trainy),2, n_ens_sel))
predfuture_ens = np.zeros((lead_time,2, n_ens_sel))

for i in range(n_ens_sel):
    pred_ens[:,:,i] = model_ens[i].predict(testX)
    predtrain_ens[:,:,i] = model_ens[i].predict(trainX)
    predfuture_ens[:,:,i] = model_ens[i].predict(futureX)


pred_mean, pred_std = mixture(pred_ens)
predtrain_mean, predtrain_std = mixture(predtrain_ens)
predfuture_mean, predfuture_std = mixture(predfuture_ens)


#%% =============================================================================
# Plot
# =============================================================================
plt.close("all")

# =============================================================================
# Loss during trianing
# =============================================================================
plt.subplots()
plt.plot(history.history['val_loss'],label = "val")
plt.plot(history.history['loss'], label= "train")
plt.legend()

# =============================================================================
# Predictions
# =============================================================================
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

std = 1.

# test
predicty_p1std = pred_mean + np.abs(pred_std)
predicty_m1std = pred_mean - np.abs(pred_std)
predicty_p2std = pred_mean + 2 * np.abs(pred_std)
predicty_m2std = pred_mean - 2 * np.abs(pred_std)


plt.fill_between(testtimey,predicty_m1std, predicty_p1std , facecolor='royalblue', alpha=0.7)
plt.fill_between(testtimey,predicty_m2std, predicty_p2std , facecolor='royalblue', alpha=0.3)
plt.plot(testtimey,pred_mean, "navy")

# train
predtrain_mean[traintimey=='2001-12-01'] = np.nan

predicttrainy_p1std = predtrain_mean +  np.abs(predtrain_std)
predicttrainy_m1std = predtrain_mean -  np.abs(predtrain_std)
predicttrainy_p2std = predtrain_mean + 2 * np.abs(predtrain_std)
predicttrainy_m2std = predtrain_mean - 2 * np.abs(predtrain_std)

plt.fill_between(traintimey,predicttrainy_m1std,predicttrainy_p1std ,facecolor='lime', alpha=0.5)
plt.fill_between(traintimey,predicttrainy_m2std,predicttrainy_p2std ,facecolor='lime', alpha=0.2)

plt.plot(traintimey, predtrain_mean, "g")

# future
predictfuturey_p1std = predfuture_mean + np.abs(predfuture_std)
predictfuturey_m1std = predfuture_mean - np.abs(predfuture_std)
predictfuturey_p2std = predfuture_mean + 2 * np.abs(predfuture_std)
predictfuturey_m2std = predfuture_mean - 2 * np.abs(predfuture_std)

plt.fill_between(futuretime, predictfuturey_m1std, predictfuturey_p1std , facecolor='orange', alpha=0.7)
plt.fill_between(futuretime, predictfuturey_m2std, predictfuturey_p2std , facecolor='orange', alpha=0.3)
plt.plot(futuretime, predfuture_mean, "darkorange")

plt.plot(timey, y, "k")

in_or_out = np.zeros((len(pred_mean)))
in_or_out[(testy>predicty_m1std) & (testy<predicty_p1std)] = 1
in_frac = np.sum(in_or_out)/len(testy)

in_or_out_train = np.zeros((len(predtrain_mean)))
in_or_out_train[(trainy>predicttrainy_m1std) & (trainy<predicttrainy_p1std)] = 1
in_frac_train = np.sum(in_or_out_train)/len(trainy)

pred_nrmse = round(nrmse(testy, pred_mean),2)

plt.title(f"train:{round(in_frac_train,2)*100}%, test:{round(in_frac*100,2)}%, NRMSE (of mean): {pred_nrmse}")
plt.grid()

# =============================================================================
# Seaonality of Standard deviations
# =============================================================================
plt.subplots()

xr_nino34 = xr.DataArray(nino34)
std_data = xr_nino34.groupby('time.month').std(dim='time')

pd_pred_std = pd.Series(data=pred_std, index = testtimey)
xr_pred_std = xr.DataArray(pd_pred_std)
std_pred = xr_pred_std.groupby('time.month').mean(dim='time')

std_data.plot()
std_pred.plot()

# =============================================================================
# plot explained variance
# =============================================================================

plot_explained_variance(testy, pred_mean, testtimey)

# =============================================================================
# Error distribution
# =============================================================================
plt.subplots()
plt.title("Error distribution")
error = pred_mean - testy

plt.hist(error, bins=16)

# =============================================================================
# ayer weight
# =============================================================================
weights = model_ens[1].get_weights()
max_w = np.max(np.abs(weights[0]))
M1=plt.matshow(weights[0], vmin=-max_w,vmax=max_w,cmap=plt.cm.seismic)
plt.colorbar(M1, extend="both")

max_w2 = np.max(np.abs(weights[1]))
M2 = plt.matshow(np.concatenate((weights[2],weights[4]),axis=1),
                 vmin=-max_w2,vmax=max_w2,cmap=plt.cm.seismic)
plt.colorbar(M2, extend="both")