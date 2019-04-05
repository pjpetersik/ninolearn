# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
import pandas as pd
import math

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy import interpolate

from ninolearn.IO.read_post import (data_reader, csv_vars)

K.clear_session()

def rmse(y, predict):
    """
    Computes the root mean square error (RMSE)

    :param y: the base line data
    :param predict: the predicted data
    :return: the RMSE
    """
    return math.sqrt(mean_squared_error(y, predict))

def nrmse(y, predict):
        """
        Computes the nromalized root mean square error (NRMSE)

        :param y: the base line data
        :param predict: the predicted data
        :return: the NRMSE
        """
        return rmse(y, predict) / (np.max([y, predict])
                                         - np.min([y, predict]))

def include_time_lag(X, max_lag=0):
    Xnew = np.copy(X[max_lag:])
    for i in range (0, max_lag):
        Xnew = np.concatenate((Xnew, X[max_lag-i-1:-i-1]), axis=1)
    return Xnew


# =============================================================================
# #%% read data
# =============================================================================
reader = data_reader(startdate='1980-01')

nino34 = reader.read_csv('nino34')
len_ts = len(nino34)
sc = np.sin(np.arange(len_ts)/12*2*np.pi)

wwv = reader.read_csv('wwv')
network = reader.read_statistic('network_metrics', variable='air',
                           dataset='NCEP', processed="anom")

pca_air = reader.read_statistic('pca', variable='air',
                           dataset='NCEP', processed="anom")
pca_u = reader.read_statistic('pca', variable='uwnd',
                           dataset='NCEP', processed="anom")
pca_v = reader.read_statistic('pca', variable='vwnd',
                           dataset='NCEP', processed="anom")


c2 = network['fraction_clusters_size_2']
c3 = network['fraction_clusters_size_3']
c5 = network['fraction_clusters_size_5']
S = network['fraction_giant_component']
H = network['corrected_hamming_distance']
T = network['global_transitivity']
C = network['avelocal_transmissivity']
L = network['average_path_length']
nwt = network['threshold']
pca1_air = pca_air['pca1']
pca2_air = pca_air['pca2']
pca3_air = pca_air['pca3']
pca1_u = pca_u['pca1']
pca2_u = pca_u['pca2']
pca3_u = pca_u['pca3']
pca1_v = pca_v['pca1']
pca2_v = pca_v['pca2']
pca3_v = pca_v['pca3']
#%% =============================================================================
# # process data
# =============================================================================
time_lag = 2
lead_time = 3

scaler = MinMaxScaler(feature_range=(-1,1))
feature_unscaled = np.stack((nino34.values, wwv.values,sc #, c2.values, c3.values, c5.values,
#                             S.values, H.values, T.values, C.values, L.values,
#                            pca1_air.values, pca2_air.values, pca3_air.values,
#                             pca1_u.values, pca2_u.values, pca3_u.values,
#                             pca1_v.values, pca2_v.values, pca3_v.values
                             ), axis=1)



Xorg = scaler.fit_transform(feature_unscaled)
X = Xorg[:-lead_time,:]

X = include_time_lag(X, max_lag=time_lag)

yorg = nino34.values
y = yorg[lead_time + time_lag:]

timey = nino34.index[lead_time + time_lag:]

train_frac = 0.67
train_end = int(train_frac * X.shape[0])
trainX, testX = X[:train_end,:], X[train_end:,:]
trainy, testy= y[:train_end], y[train_end:]
traintimey, testtimey = timey[:train_end], timey[train_end:]
# =============================================================================
# # neural network
# =============================================================================
model = Sequential()

model.add(Dense(32, input_dim=X.shape[1],activation='relu', kernel_regularizer=regularizers.l2(0.005)))
model.add(Dropout(0.2))
model.add(Dense(8, input_dim=X.shape[1],activation='relu', kernel_regularizer=regularizers.l2(0.005)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mse'])

es = EarlyStopping(monitor='val_loss',
                          min_delta=0.0,
                          patience=40,
                          verbose=0, mode='auto')

history = model.fit(trainX, trainy, epochs=500, batch_size=20,verbose=0,
                    shuffle=True, callbacks=[es],
                    validation_data=(testX, testy))

predicty = model.predict(testX)

score = nrmse(testy, predicty[:,0])

print(f"NRMSE: {score}")
corr = np.corrcoef(testy, predicty[:,0])[0,1]
print(f"R^2: {corr}")
#%% =============================================================================
# # plot
# =============================================================================
plt.close("all")
plt.plot(testtimey,predicty, "r")
from ninolearn.plot.evaluation  import plot_explained_variance
#for _ in range(100):
#    predicty = model.predict(np.clip(testX+np.random.uniform(-0.5,0.5, size=testX.shape), -1,1))
#    plt.plot(testtimey,predicty, "r")

plt.plot(testtimey,testy, "k")
plt.subplots()
plt.plot(history.history['val_loss'],label = "val")
plt.plot(history.history['loss'], label= "train")
plt.legend()

plot_explained_variance(testy, predicty[:,0], testtimey)