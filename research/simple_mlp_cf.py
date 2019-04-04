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

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy import interpolate
from scipy.signal import detrend
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
reader = data_reader(startdate="1950-01")

nino34 = reader.read_csv('nino34')

#wwv = reader.read_csv('wwv')
network = reader.read_statistic('network_metrics', variable='air_daily',
                           dataset='NCEP', processed="anom")

c2 = network['fraction_clusters_size_2']
c3 = network['fraction_clusters_size_3']
c5 = network['fraction_clusters_size_5']
S = network['fraction_giant_component']

# =============================================================================
# # process data
# =============================================================================
time_lag = 24
lead_time = 6
nino_treshold = 0.8
scaler = MinMaxScaler(feature_range=(-1,1))
feature_unscaled = np.stack((c2.values, c3.values, S.values, nino34.values), axis=1)
Xorg = scaler.fit_transform(feature_unscaled)
X = Xorg[:-lead_time,:]

X = include_time_lag(X, max_lag=time_lag)

yorg = np.zeros_like(nino34.values,dtype=int) + 1
yorg[nino34.values>nino_treshold] = 2
yorg[nino34.values<-nino_treshold] = 0

#%%
y = yorg[lead_time + time_lag:]
timey = nino34.index[lead_time + time_lag:]
nino34y = nino34[lead_time + time_lag:]


train_frac = 0.8
train_end = int(train_frac * X.shape[0])
trainX, testX = X[:train_end,:], X[train_end:,:]
trainy, testy= y[:train_end], y[train_end:]
traintimey, testtimey = timey[:train_end], timey[train_end:]

nino34testy = nino34y[train_end:]
# =============================================================================
# # neural network
# =============================================================================
model = Sequential()

model.add(Dense(32, input_dim=X.shape[1],activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(32, input_dim=X.shape[1],activation='sigmoid'))
model.add(Dense(16, input_dim=X.shape[1],activation='relu'))
model.add(Dense(3, activation='softmax'))

optimizer = Adam(lr=0.5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss',
                          min_delta=0.0,
                          patience=10,
                          verbose=0, mode='auto')

history = model.fit(trainX, trainy, epochs=100, batch_size=10,verbose=0,
                    shuffle=True, #callbacks=[es],
                    validation_data=(testX, testy))

predicty = model.predict(testX)


#%% =============================================================================
# # plot
# =============================================================================
plt.close("all")
from ninolearn.plot.nino_timeseries import nino_background

plt.plot(testtimey, predicty[:,0], label="La Nina", c="blue")
#plt.plot(testtimey, predicty[:,1], label="Neutral", c="k")
plt.plot(testtimey, predicty[:,2], label="El Nino", c="k")
nino_background(nino34testy, nino_treshold=nino_treshold)
plt.ylim(0,1)
plt.legend()

#nino_background(nino_data=ninoy, nino_treshold=1)

plt.subplots()
plt.plot(history.history['loss'],label = "train")
plt.plot(history.history['val_loss'], label= "val")
plt.legend()