# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
import pandas as pd
import math

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv1D
from keras.layers import LSTM
from keras.layers import Dropout, GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy import interpolate
from scipy.signal import detrend
from ninolearn.IO.read_post import (data_reader, csv_vars)
from ninolearn.learn.augment import window_warping

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

def nino_to_category(nino, categories=3, threshold=None):
    if threshold == None:
        sorted_arr = np.sort(nino)
        n = len(sorted_arr)
        n_cat = n//categories
        bounds = np.zeros(categories+1)

        for i in range(1,categories):
            bounds[i] = sorted_arr[i*n_cat]
        bounds[0] = sorted_arr[0] -1
        bounds[-1] = sorted_arr[-1]+1

        nino_cat = np.zeros_like(nino, dtype=int) + categories

        for j in range(categories):
            nino_cat[(nino>bounds[j]) & (nino<=bounds[j+1])] = j

        assert (nino_cat != categories).all()
    else:
        nino_cat = np.zeros_like(nino, dtype=int) + 1
        nino_cat[nino>threshold] = 2
        nino_cat[nino<-threshold] = 0
    return nino_cat

def plot_confMat(y, pred, labels):

    cm = confusion_matrix(y, pred).T
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin = 1/len(labels), vmax = 1)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels,
           yticklabels=labels,
           title='Confusion Matrix',
           xlabel='True label',
           ylabel='Predicted label')

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black" if cm[i, j] > thresh else "black")
    fig.tight_layout()



# =============================================================================
# #%% read data
# =============================================================================
reader = data_reader(startdate="1950-01")

nino34 = reader.read_csv('nino34')

#wwv = reader.read_csv('wwv')
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
time_lag = 6
lead_time = 6
#classes = 3
threshold = 0.5
class_labels = ['La nina', 'Neutral','El Nino']
#class_labels = ['La nina', 'negative neutral', 'postive neutral', 'El Nino']
#class_labels = ['Strong La nina', 'Weak La nina',
                #'Neutral','Weak El Nino', 'Strong El Nino']

feature_unscaled = np.stack((nino34.values, nwt.values, #c2.values, c3.values, c5.values,
                             S.values, H.values, T.values, C.values, L.values,
                             pca1_air.values, pca2_air.values, pca3_air.values,
                             pca1_u.values, pca2_u.values, pca3_u.values,
                             pca1_v.values, pca2_v.values, pca3_v.values
                             ), axis=1)

scaler = MinMaxScaler(feature_range=(-1,1))
#just_one =  nino_to_category(np.roll(nino34.values,-12), 2.)
#feature_unscaled = just_one.reshape(len(just_one),1)
Xorg = scaler.fit_transform(feature_unscaled)

X = Xorg[:-lead_time,:]

X = include_time_lag(X, max_lag=time_lag)
Xfull = include_time_lag(Xorg, max_lag=time_lag)

yorg = nino_to_category(nino34.values, threshold=threshold)
yshift_predict = yorg[time_lag:-lead_time]
#%%
y = yorg[lead_time + time_lag:]
timey = nino34.index[lead_time + time_lag:]
nino34y = nino34[lead_time + time_lag:]

train_frac = 0.6
train_end = int(train_frac * X.shape[0])
trainX, testX = X[:train_end,:], X[train_end:,:]
trainy, testy= y[:train_end], y[train_end:]
traintimey, testtimey = timey[:train_end], timey[train_end:]

nino34trainy, nino34testy = nino34y[:train_end], nino34y[train_end:]
# =============================================================================
# # neural network
# =============================================================================
model = Sequential()

model.add(Dense(32, input_dim=X.shape[1],activation='relu', kernel_regularizer=regularizers.l1(0.003)))
model.add(Dropout(0.2))
model.add(Dense(16, input_dim=X.shape[1],activation='relu', kernel_regularizer=regularizers.l1(0.003)))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_acc',
                          min_delta=0.0,
                          patience=20,
                          verbose=0, mode='auto')

history = model.fit(trainX, trainy, epochs=250, batch_size=20,verbose=0,
                    shuffle=True, callbacks=[es],
                    validation_data=(testX, testy))


#for _ in range(10):
#    #trainX = window_warping(trainX, strength=0.2, window_size=[1,12])
#    nino = window_warping(nino34trainy.values, strength=0.5, window_size=[1,12])
#    trainy = nino_to_category(nino, nino_treshold)
#
#    history = model.fit(trainX, trainy, epochs=25, batch_size=20,verbose=0,
#                        shuffle=True, callbacks=[es],
#                        validation_data=(testX, testy))
#
#    print(history.history['val_loss'][-1])
#    print(history.history['val_acc'][-1])
#    print(f"Epochs: {history.epoch[-1]}")


# =============================================================================
# Predict
# =============================================================================

predicty = model.predict(testX)

#%% =============================================================================
# # plot
# =============================================================================
plt.close("all")
from ninolearn.plot.nino_timeseries import nino_background
plt.figure(1)
plt.plot(testtimey, predicty[:,0], label="La Nina", c="blue", marker="x")
plt.plot(testtimey, predicty[:,1], label="Neutral", c="k")
plt.plot(testtimey, predicty[:,2], label="El Nino", c="r", marker="x")

nino_background(nino34testy, nino_treshold=0.5)
plt.ylim(0,1)
plt.legend()


plt.figure(2)
plt.title("Loss")
plt.plot(history.history['loss'],label = "train")
plt.plot(history.history['val_loss'], label= "val")
plt.legend()

plt.figure(3)
plt.title("Accuracy")
plt.plot(history.history['acc'],label = "train")
plt.plot(history.history['val_acc'], label= "val")
plt.legend()

predict_cat = np.argmax(predicty,axis=1)
prec, recall, fscore, support = precision_recall_fscore_support(testy, predict_cat)

plot_confMat(testy, predict_cat, class_labels)
plot_confMat(testy, yshift_predict[train_end:], class_labels)
