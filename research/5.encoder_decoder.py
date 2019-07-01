# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import keras.backend as K
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Input, concatenate
from keras.layers import Dropout, GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers

from sklearn.preprocessing import StandardScaler

from ninolearn.IO.read_post import data_reader
from ninolearn.utils import scale
import matplotlib.animation as animation
from IPython.display import HTML
#%%
#free memory from old sessions
K.clear_session()

#read data
reader = data_reader(startdate='1951-01', enddate='2018-12')#, lat_max=60,lon_min=30)
#ssh = reader.read_netcdf('sshg', dataset='GODAS', processed='anom')
#ssh = reader.read_netcdf('zos', dataset='ORAS4', processed='anom')
sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom')
#sat = reader.read_netcdf('air', dataset='NCEP', processed='anom')
#uwnd = reader.read_netcdf('uwnd', dataset='NCEP', processed='anom')
#taux = reader.read_netcdf('taux', dataset='NCEP', processed='anom')
nino34 = reader.read_csv('nino3.4S')

# select
feature = sst
label = sst

# preprocess data
feature_unscaled = feature.values.reshape(feature.shape[0],-1)
label_unscaled = label.values.reshape(label.shape[0],-1)



scaler_f = StandardScaler()
Xorg = scaler_f.fit_transform(feature_unscaled)

scaler_l = StandardScaler()
yorg = scaler_l.fit_transform(label_unscaled)

Xall = np.nan_to_num(Xorg)
yall = np.nan_to_num(yorg)

lead = 3

if lead == 0:
    y = yall
    X = Xall
else:
    y = yall[lead:]
    X = Xall[:-lead]

n_obs = len(feature)

train_end = int(0.65 * n_obs)
val_end = int(0.85 * n_obs)

X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]



# =============================================================================
# ENCODER-DECODER
# =============================================================================
# this is the size of our encoded representations
encoding_dim = 16 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
hidden_dim = 128
l1 = 0.0
l2 = 0.0001

# encoder
inputs = Input(shape=(X.shape[1],))
h = GaussianNoise(0.2)(inputs)


h = Dense(hidden_dim, activation='relu',
          kernel_regularizer=regularizers.l1_l2(l1, l2),
          kernel_initializer="random_normal")(h)

latent = Dense(encoding_dim, activation='linear',
               kernel_regularizer=regularizers.l1_l2(l1, l2))(h)

encoder = Model(inputs, latent, name='encoder')

# decoder
latent_inputs = Input(shape=(encoding_dim,))
#h = Dense(hidden_dim, activation='relu',
#          kernel_regularizer=regularizers.l1_l2(l1, l2))(latent_inputs)

decoded = Dense(X.shape[1], activation='linear',
                kernel_regularizer=regularizers.l1_l2(l1, l2))(latent_inputs)

decoder = Model(latent_inputs, decoded, name='decoder')

# autoencoder
autoencoder = Model(inputs, decoder(encoder(inputs)))


optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False)
es = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0, patience=50, verbose=0,
                   mode='min', restore_best_weights=True)

autoencoder.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

autoencoder.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=50,
                shuffle=True, callbacks=[es])

#%% Predictions
predy = autoencoder.predict(Xall)

label_unscaled_decoded = scaler_l.inverse_transform(predy)

label_decoded = xr.zeros_like(label)
label_decoded.values = label_unscaled_decoded.reshape((Xall.shape[0], label.shape[1],label.shape[2]))

latent_variables = encoder.predict(Xall)

print(f"Score: {autoencoder.evaluate(X_test, y_test)[1]}")
#%%
plt.close("all")


# NINO3.4 Predicted vs. Observation
nino34_pred = label_decoded.loc[dict(lat=slice(-5, 5), lon=slice(210, 240))].mean(dim="lat").mean(dim='lon')

plt.subplots()
plt.plot(nino34_pred.time[lead+val_end:], nino34.values[lead+val_end:])
plt.plot(nino34_pred.time[lead+val_end:], nino34_pred.values[val_end:-lead])
print(np.corrcoef(nino34.values[lead:], nino34_pred.values[:-lead]))



# A static plot
#frame = 442-lead
frame = 410-lead
fig1, ax1 = plt.subplots(3, 1)

vmin = -1.3
vmax = 1.3
ax1[0].imshow(label[frame], origin='lower', vmin=vmin, vmax=vmax)
ax1[1].imshow(label[frame+lead], origin='lower', vmin=vmin, vmax=vmax)
ax1[2].imshow(label_decoded[frame], origin='lower', vmin=vmin, vmax=vmax)


fig, ax = plt.subplots(3, 1, figsize=(6,7), squeeze=False)
true = label[val_end+lead]
pred = label_decoded[val_end]

true_im = ax[0,0].imshow(true, origin='lower', vmin=-1.3, vmax=1.3)
pred_im = ax[1,0].imshow(pred, origin='lower', vmin=-1.3, vmax=1.3)
title = ax[0,0].set_title('')


ax[2,0].plot(nino34_pred.time[lead+val_end:], nino34.values[lead+val_end:])
ax[2,0].plot(nino34_pred.time[lead+val_end:], nino34_pred.values[val_end:-lead])
ax[2,0].set_ylim(-3,3)
ax[2,0].set_xlim(nino34_pred.time[lead+val_end].values, nino34_pred.time[-1].values)

vline = ax[2,0].plot([nino34_pred.time[lead+val_end].values,nino34_pred.time[lead+val_end].values], [-10,10], color='k')


def update(data):
    true_im.set_data(data[0])
    pred_im.set_data(data[1])
    title_str = np.datetime_as_string(data[0].time.values)[:10]
    title.set_text(title_str)

    vline[0].set_data([data[2], data[2]],[-10,10])

def data_gen():
    k=0
    kmax = len(y_test)
    while k<kmax:
        yield label[val_end+lead+k], label_decoded[val_end+k], nino34_pred.time[lead+val_end+k].values
        k+=1

ani = animation.FuncAnimation(fig, update, data_gen, interval=300)


#%%
corrM = np.zeros(len(predy[0,:]))

for i in range(len(predy[0,:])):
    corrM[i] = np.corrcoef(predy[:,i], yall[:,i])[0,1]


corrM=corrM.reshape((label.shape[1:]))
fig, ax = plt.subplots(figsize=(8,2))

vmin = 0.
vmax = 1.
C=ax.imshow(corrM, origin='lower', vmin=vmin, vmax=vmax)
plt.colorbar(C)