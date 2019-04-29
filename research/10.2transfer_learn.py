"""
In this script, I want to build a deep ensemble that is first trained on the GFDL data
and then trained on the observations
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from os.path import join, exists
from os import remove, mkdir

import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import model_from_json, load_model, save_model
from keras.regularizers import l1_l2, l1

from sklearn.preprocessing import StandardScaler

from ninolearn.IO.read_post import data_reader
from ninolearn.plot.evaluation  import plot_explained_variance
from ninolearn.learn.evaluation import nrmse, rmse, inside_fraction
from ninolearn.learn.mlp import include_time_lag
from ninolearn.learn.losses import nll_gaussian
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
reader = data_reader(startdate='1981-01', enddate='2018-12')

nino34 = reader.read_csv('nino3.4M')
wwv = reader.read_csv('wwv')
iod = reader.read_csv('iod')

#PCA data
pca_air = reader.read_statistic('pca', variable='air',
                           dataset='NCEP', processed="anom")

pca2_air = pca_air['pca2']

# Network metics
nwm_ssh = reader.read_statistic('network_metrics', variable='sshg',
                           dataset='GODAS', processed="anom")

c2_ssh = nwm_ssh['fraction_clusters_size_2']
S_ssh = nwm_ssh['fraction_giant_component']
H_ssh = nwm_ssh['corrected_hamming_distance']
T_ssh = nwm_ssh['global_transitivity']
C_ssh = nwm_ssh['avelocal_transmissivity']
L_ssh = nwm_ssh['average_path_length']

nwm_air = reader.read_statistic('network_metrics', variable='air',
                           dataset='NCEP', processed="anom")

S_air = nwm_air['fraction_giant_component']
T_air = nwm_air['global_transitivity']

len_ts = len(nino34)
sc = np.cos(np.arange(len_ts)/12*2*np.pi)
yr =  np.arange(len_ts) % 12

#%% =============================================================================
# # process data
# =============================================================================
time_lag = 12
lead_time = 12

feature_unscaled = np.stack((nino34,  sc,# yr,
                            iod,
                            c2_ssh, S_ssh, H_ssh, T_ssh, C_ssh, L_ssh,
#                             S_air, T_air,
#                             pca2_air
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
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)

es = EarlyStopping(monitor='val_nll_gaussian',
                              min_delta=0.0,
                              patience=10,
                              verbose=0,
                              mode='min',
                              restore_best_weights=True)

rejected = True
while rejected:
    # load  model
    path_pretrained = join(modeldir, f"model{lead_time}.h5")
    model = load_model(path_pretrained)

    model.get_layer('dense_1').kernel_regularizer = l1_l2(0.1, 0.1)
    model.get_layer('dense_2').kernel_regularizer = l1_l2(0.1, 0.1)
    model.get_layer('dense_3').kernel_regularizer = l1_l2(0.1, 0.1)
    model.save("temp_path.h5")
    model = load_model("temp_path.h5")
    remove("temp_path.h5")

    model.compile(loss=nll_gaussian, optimizer=optimizer, metrics=[nll_gaussian])

    eva = model.evaluate(testX, testy)
    print(eva)

    print_header("Train")
    history = model.fit(trainX, trainy, epochs=300, batch_size=1,verbose=1,
                        shuffle=True, callbacks=[es],
                        validation_data=(testX, testy))

    mem_pred = model.predict(trainX)
    mem_mean = mem_pred[:,0]
    mem_std = np.abs(mem_pred[:,1])

    in_frac = inside_fraction(trainy, mem_mean, mem_std)

    if in_frac > 0.85 or in_frac < 0.45:
        print_header("Reject this model. Unreasonable stds.")


    elif rmse(trainy, mem_mean)>1.:
        print_header(f"Reject this model. Unreasonalble rmse of {rmse(trainy, mem_mean)}")

    else:
        rejected = False

#%% =============================================================================
# Save
# =============================================================================

if not exists(modeldir):
    mkdir(modeldir)

path_trained = join(modeldir, f"model2{lead_time}.h5")
save_model(model, path_trained, include_optimizer=False)

print("Saved model to disk")


#%% =============================================================================
# predict
# =============================================================================
predtest = model.predict(testX)
pred_mean, pred_std = predtest[:,0], predtest[:,1]

predtrain = model.predict(trainX)
predtrain_mean, predtrain_std = predtrain[:,0], predtrain[:,1]

predfuture = model.predict(futureX)
predfuture_mean, predfuture_std = predfuture[:,0], predfuture[:,1]


#%% =============================================================================
# Plot
# =============================================================================
plt.close("all")

# =============================================================================
# Loss during trianing
# =============================================================================
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].plot(history.history['val_loss'],label = "val")
ax[0].plot(history.history['loss'], label= "train")
ax[0].legend()

ax[1].plot(history.history['val_nll_gaussian'],label = "val")
ax[1].plot(history.history['nll_gaussian'], label= "train")
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
# layer weight
# =============================================================================
pretrained_model = load_model(path_pretrained)
weights_pre = pretrained_model.get_weights()
weights = model.get_weights()

weight_diff =weights[0]# weights_pre[0] -

max_w = np.max(np.abs(weights[0]))
M1=plt.matshow(weight_diff, vmin=-max_w,vmax=max_w,cmap=plt.cm.seismic)
plt.colorbar(M1, extend="both")

max_w2 = np.max(np.abs(weights[1]))
M2 = plt.matshow(np.concatenate((weights[2],weights[4]),axis=1),
                 vmin=-max_w2,vmax=max_w2,cmap=plt.cm.seismic)
plt.colorbar(M2, extend="both")