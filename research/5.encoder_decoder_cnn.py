# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import keras.backend as K
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Input, concatenate, Conv2D, Flatten, Reshape, Conv2DTranspose, Activation
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
reader = data_reader(startdate='1981-01', enddate='2018-12')
#ssh = reader.read_netcdf('sshg', dataset='GODAS', processed='anom')
sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom')
#sat = reader.read_netcdf('air', dataset='NCEP', processed='anom')
nino34 = reader.read_csv('nino3.4S')

# select
feature = sst
label = sst

# preprocess data
shape = feature.shape
feature_unscaled = feature.values.reshape((shape[0], shape[1],shape[2], 1))
label_unscaled = label.values.reshape((shape[0], shape[1],shape[2], 1))

Xall = np.nan_to_num(feature_unscaled)
yall = np.nan_to_num(label_unscaled)

lead = 6

if lead == 0:
    y = yall
    X = Xall
else:
    y = yall[lead:]
    X = Xall[:-lead]

#%%
n_obs = len(feature)

train_end = int(0.7 * n_obs)
val_end = int(0.8 * n_obs)

X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

# =============================================================================
# ENCODER-DECODER
# =============================================================================
# this is the size of our encoded representations
encoding_dim = 8

kernel_size = 32
filters = 4

l1 = 0.00
l2 = 0.00

# ENCODER
inputs = Input(shape=(X.shape[1], X.shape[2], 1))

h = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=5,
               activation='relu',
               padding='same')(inputs)

# shape of the convolved image
shape2 = K.int_shape(h)

h = Flatten()(h)
latent = Dense(encoding_dim, activation='linear',
               kernel_regularizer=regularizers.l1_l2(l1, l2))(h)

encoder = Model(inputs, latent, name='encoder')

# DECODER
latent_inputs = Input(shape=(encoding_dim,))
h = Dense(shape2[1] * shape2[2] * shape2[3])(latent_inputs)
h = Reshape((shape2[1], shape2[2], shape2[3]))(h)

h = Conv2DTranspose(filters=filters,
               kernel_size=kernel_size,
               strides=5,
               activation='relu',
               padding='same')(h)

outputs = Conv2DTranspose(filters=1,
               kernel_size=kernel_size,
               strides=1,
               activation='linear',
               padding='same')(h)

#outputs = Activation('linear', name='decoder_output')(h)
decoder = Model(latent_inputs, outputs, name='decoder')

# AUTOENCODER
autoencoder = Model(inputs, decoder(encoder(inputs)))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False)
es = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0, patience=10, verbose=0,
                   mode='min', restore_best_weights=True)

autoencoder.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

autoencoder.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=100,
                shuffle=True, callbacks=[es])

#%% Predictions
label_unscaled_decoded = autoencoder.predict(Xall)

label_decoded = xr.zeros_like(label)
label_decoded.values = label_unscaled_decoded.reshape((Xall.shape[0], label.shape[1],label.shape[2]))
label_decoded.values[np.isnan(label.values)] = np.nan

latent_variables = encoder.predict(Xall)

print(f"Score: {autoencoder.evaluate(X_test, y_test)[1]}")
#%%
plt.close("all")

#frame = 442-lead
frame = 410-lead
fig, ax = plt.subplots(3, 1)

vmin = -1.3
vmax = 1.3
ax[0].imshow(label[frame], origin='lower', vmin=vmin, vmax=vmax)
ax[1].imshow(label[frame+lead], origin='lower', vmin=vmin, vmax=vmax)
ax[2].imshow(label_decoded[frame], origin='lower', vmin=vmin, vmax=vmax)

#%%

fig, ax = plt.subplots(2, 1)

true = label[val_end+lead]
pred = label_decoded[val_end]

true_im = ax[0].imshow(true, origin='lower', vmin=-1.3, vmax=1.3)
pred_im = ax[1].imshow(pred, origin='lower', vmin=-1.3, vmax=1.3)

def update(data):
    true_im.set_data(data[0])
    pred_im.set_data(data[1])

def data_gen():
    k=0
    kmax = len(y_test)
    while k<kmax:
        yield label[val_end+lead+k], label_decoded[val_end+k]
        k+=1
ani = animation.FuncAnimation(fig, update, data_gen, interval=200)

#%%

nino34_pred = label_decoded.loc[dict(lat=slice(-5, 5), lon=slice(210, 240))].mean(dim="lat").mean(dim='lon')

plt.subplots()
plt.plot(nino34_pred.time[lead+val_end:], nino34.values[lead+val_end:])
plt.plot(nino34_pred.time[lead+val_end:], nino34_pred.values[val_end:-lead])
print(np.corrcoef(nino34.values[lead:],nino34_pred.values[:-lead]))