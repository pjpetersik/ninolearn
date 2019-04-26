"""
In this script, I want to build a deep ensemble that is first trained on the GFDL data
and then trained on the observations
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from os.path import exists, join
from os import mkdir

import keras.backend as K
from keras.models import Sequential, Model, save_model
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

# PCA data
pca_air = reader.read_statistic('pca', variable='tas',
                           dataset='GFDL-CM3', processed="anom")

pca1_air = pca_air['pca1']
pca2_air = pca_air['pca2']
pca3_air = pca_air['pca3']

# Network metics
nwm_ssh = reader.read_statistic('network_metrics', 'zos',
                                        dataset='GFDL-CM3',
                                        processed='anom')

c2 = nwm_ssh['fraction_clusters_size_2']
c3 = nwm_ssh['fraction_clusters_size_3']
c5 = nwm_ssh['fraction_clusters_size_5']
S = nwm_ssh['fraction_giant_component']
H = nwm_ssh['corrected_hamming_distance']
T = nwm_ssh['global_transitivity']
C = nwm_ssh['avelocal_transmissivity']
L = nwm_ssh['average_path_length']
nwt = nwm_ssh['threshold']

# artificiial data
len_ts = len(nino34)
sc = np.cos(np.arange(len_ts)/12*2*np.pi)
yr =  np.arange(len_ts) % 12

#%% =============================================================================
# # process data
# =============================================================================
time_lag = 12
lead_time = 9

feature_unscaled = np.stack((nino34, sc, yr,
                             c2, c3, c5, S, H, T, C, L,
                             pca1_air, pca2_air, pca3_air), axis=1)


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

train_frac = 0.8
train_end = int(train_frac * X.shape[0])

trainX, testX = X[:train_end,:], X[train_end:,:]
trainy, testy= y[:train_end], y[train_end:]
traintimey, testtimey = timey[:train_end], timey[train_end:]


#%% =============================================================================
# # neural network
# =============================================================================
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

es = EarlyStopping(monitor='val_loss',
                              min_delta=0.0,
                              patience=10,
                              verbose=0,
                              mode='min',
                              restore_best_weights=True)

l1 = 0.01
l2 = 0.1

l1_out = 0.0
l2_out = 0.01

rejected = True
while rejected:

    # define the model
    inputs = Input(shape=(trainX.shape[1],))
    h = GaussianNoise(0.2)(inputs)

    h = Dense(8, activation='relu',
              kernel_regularizer=regularizers.l1_l2(l1, l2))(h)
    h = Dropout(0.2)(h)

    mu = Dense(1, activation='linear', kernel_regularizer=regularizers.l1_l2(l1_out, l2_out))(h)
    sigma = Dense(1, activation='softplus', kernel_regularizer=regularizers.l1_l2(l1_out, l2_out))(h)

    outputs = concatenate([mu, sigma])

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=nll_gaussian, optimizer=optimizer)

    print_header("Train")
    history = model.fit(trainX, trainy, epochs=300, batch_size=10,verbose=1,
                        shuffle=True, callbacks=[es],
                        validation_data=(testX, testy))

    mem_pred = model.predict(trainX)
    mem_mean = mem_pred[:,0]
    mem_std = np.abs(mem_pred[:,1])

    in_frac = inside_fraction(trainy, mem_mean, mem_std)

    if in_frac > 0.8 or in_frac < 0.55:
        print_header("Reject this model. Unreasonable stds.")


    elif rmse(trainy, mem_mean)>1.:
        print_header(f"Reject this model. Unreasonalble rmse of {rmse(trainy, mem_mean)}")


    elif np.min(history.history["loss"])>1:
        print_header("Reject this model. High minimum loss.")

    else:
        rejected = False


#%% =============================================================================
# Save
# =============================================================================

if not exists(modeldir):
    mkdir(modeldir)

path_h5 = join(modeldir, f"model{lead_time}.h5")
save_model(model, path_h5, include_optimizer=False)

print("Saved model to disk")

#%% =============================================================================
# predict
# =============================================================================
predtest = model.predict(testX)
pred_mean, pred_std = predtest[:,0], predtest[:,1]

predtrain = model.predict(trainX)
predtrain_mean, predtrain_std = predtrain[:,0], predtrain[:,1]

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

#plt.xlim(timey[0],futuretime[-1])
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


predicttrainy_p1std = predtrain_mean +  np.abs(predtrain_std)
predicttrainy_m1std = predtrain_mean -  np.abs(predtrain_std)
predicttrainy_p2std = predtrain_mean + 2 * np.abs(predtrain_std)
predicttrainy_m2std = predtrain_mean - 2 * np.abs(predtrain_std)

plt.fill_between(traintimey,predicttrainy_m1std,predicttrainy_p1std ,facecolor='lime', alpha=0.5)
plt.fill_between(traintimey,predicttrainy_m2std,predicttrainy_p2std ,facecolor='lime', alpha=0.2)


plt.plot(traintimey, predtrain_mean, "g")

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
# layer weight
# =============================================================================
weights = model.get_weights()
max_w = np.max(np.abs(weights[0]))
M1=plt.matshow(weights[0], vmin=-max_w,vmax=max_w,cmap=plt.cm.seismic)
plt.colorbar(M1, extend="both")

max_w2 = np.max(np.abs(weights[1]))
M2 = plt.matshow(np.concatenate((weights[2],weights[4]),axis=1),
                 vmin=-max_w2,vmax=max_w2,cmap=plt.cm.seismic)
plt.colorbar(M2, extend="both")
