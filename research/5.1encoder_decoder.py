# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.animation as animation

import keras.backend as K

from sklearn.preprocessing import StandardScaler

from ninolearn.IO.read_post import data_reader
from ninolearn.learn.encoderDecoder import EncoderDecoder
from ninolearn.pathes import ed_model_dir
from ninolearn.utils import print_header

#%%free memory from old sessions
K.clear_session()
# =============================================================================
# Data
# =============================================================================
#read data
reader = data_reader(startdate='1959-11', enddate='2017-12')
sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom').rolling(time=3).mean()[2:]
nino34 = reader.read_csv('nino3.4S')[2:]
#%%
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
for lead in [0, 3, 6, 9, 12, 15]:
    print_header(f'Lead time: {lead} month')

    y = yall[lead+shift:]
    X = Xall[:-lead-shift]
    timey = nino34.index[lead+shift:]

    for decade in [60, 70, 80, 90, 100, 110]:
        print_header(f'Test period: {1902+decade}-01-01 till {1911+decade}-12-01')
        K.clear_session()

        test_indeces = test_indeces = (timey>=f'{1902+decade}-01-01') & (timey<=f'{1911+decade}-12-01')
        train_indeces = np.invert(test_indeces)

        trainX, trainy, traintimey = X[train_indeces,:], y[train_indeces,:], timey[train_indeces]
        testX, testy, testtimey = X[test_indeces,:], y[test_indeces,:], timey[test_indeces]

        # =============================================================================
        # Model
        # =============================================================================

        model = EncoderDecoder()

        model.set_parameters(neurons=(128, 16), dropout=0.2, noise=0.2, noise_out=0.2,
                 l1_hidden=0.0001, l2_hidden=0.0001, l1_out=0.0001, l2_out=0.0001, batch_size=100,
                 lr=0.01, n_segments=5, n_members_segment=1, patience = 40, epochs=500, verbose=0)

        model.fit(trainX, trainy)
        dir_name = f'ed_ensemble_decade{decade}_lead{lead}'
        model.save(location=ed_model_dir, dir_name=dir_name)


#%% =============================================================================
# Prediction
# =============================================================================
predy, pred_ens = model.predict(testX)

predy_decoded = xr.zeros_like(label[lead+shift:,:,:][test_indeces])
predy_decoded.values = scaler_l.inverse_transform(predy).reshape((testX.shape[0], label.shape[1], label.shape[2]))

nino34_pred = predy_decoded.loc[dict(lat=slice(-5, 5), lon=slice(190, 240))].mean(dim="lat").mean(dim='lon').rolling(time=3).mean()

print(np.corrcoef(nino34[lead+shift:][test_indeces][2:], nino34_pred[2:]))






#%% =============================================================================
# Plot
# =============================================================================
plt.close("all")

# =============================================================================
# # NINO3.4 Predicted vs. Observation
# =============================================================================
plt.subplots()
plt.plot(nino34_pred.time, nino34_pred)
plt.plot(nino34_pred.time, nino34[lead+shift:][test_indeces])


# =============================================================================
# Animation
# =============================================================================
fig, ax = plt.subplots(3, 1, figsize=(6,7), squeeze=False)

vmin = -3
vmax = 3
true = label[lead+shift:][test_indeces]
pred = predy_decoded

true_im = ax[0,0].imshow(true[0], origin='lower', vmin=vmin, vmax=vmax, cmap=plt.cm.bwr)
pred_im = ax[1,0].imshow(pred[0], origin='lower', vmin=vmin, vmax=vmax, cmap=plt.cm.bwr)
title = ax[0,0].set_title('')


ax[2,0].plot(nino34_pred.time, nino34.values[lead+shift:][test_indeces])
ax[2,0].plot(nino34_pred.time, nino34_pred.values)
ax[2,0].set_ylim(-3,3)
ax[2,0].set_xlim(nino34_pred.time.values[0], nino34_pred.time[-1].values)

vline = ax[2,0].plot([nino34_pred.time.values[0], nino34_pred.time[0].values], [-10,10], color='k')


def update(data):
    true_im.set_data(data[0])
    pred_im.set_data(data[1])
    title_str = np.datetime_as_string(data[0].time.values)[:10]
    title.set_text(title_str)

    vline[0].set_data([data[2], data[2]],[-10,10])

def data_gen():
    k=0
    kmax = len(testy)
    while k<kmax:
        yield true[k], pred[k], nino34_pred.time[k].values
        k+=1

ani = animation.FuncAnimation(fig, update, data_gen, interval=100)

# =============================================================================
# Correlation
# =============================================================================
corrM = np.zeros(len(predy[0,:]))

for i in range(len(predy[0,:])):
    corrM[i] = np.corrcoef(predy[:,i], testy[:,i])[0,1]


corrM=corrM.reshape((label.shape[1:]))
fig, ax = plt.subplots(figsize=(8,2))

vmin = 0.
vmax = 1.
C=ax.imshow(corrM, origin='lower', vmin=vmin, vmax=vmax)
plt.colorbar(C)
