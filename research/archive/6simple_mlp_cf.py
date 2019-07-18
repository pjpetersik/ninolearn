import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers

from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler

from ninolearn.IO.read_post import data_reader
from ninolearn.plot.evaluation import plot_confMat
from ninolearn.utils import nino_to_category, include_time_lag

K.clear_session()

# =============================================================================
# #%% read data
# =============================================================================
reader = data_reader(startdate="1981-01")

nino4 = reader.read_csv('nino4M')
nino34 = reader.read_csv('nino3.4M')
nino12 = reader.read_csv('nino1+2M')
nino3 = reader.read_csv('nino3M')

len_ts = len(nino34)
sc = np.cos(np.arange(len_ts)/12*2*np.pi)
yr =  np.arange(len_ts) % 12
yr3 = np.arange(len_ts) % 36
yr4 = np.arange(len_ts) % 48
yr5 = np.arange(len_ts) % 60


wwv = reader.read_csv('wwv')
network = reader.read_statistic('network_metrics', variable='air_daily',
                           dataset='NCEP', processed="anom")

network_ssh = reader.read_statistic('network_metrics', variable='sshg',
                           dataset='GODAS', processed="anom")

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

c2ssh = network_ssh['fraction_clusters_size_2']

#%% =============================================================================
# # process data
# =============================================================================
time_lag = 6
lead_time = 12

classes = 3
class_labels = ['La nina', 'Neutral','El Nino']
threshold = 0.5

train_frac = 0.7

feature_unscaled = np.stack((nino34.values, #nino12.values , nino3.values, nino4.values,
                             wwv.values, sc,  c2ssh.values, # yr,  nwt.values,
#                             c2.values, ,c3.values, c5.values,
#                             S.values, H.values, T.values, C.values, L.values,
#                             pca1_air.values, pca2_air.values, pca3_air.values,
#                             pca1_u.values, pca2_u.values, pca3_u.values,
#                             pca1_v.values, pca2_v.values, pca3_v.values
                             ), axis=1)

scaler = MinMaxScaler(feature_range=(-1,1))
Xorg = scaler.fit_transform(feature_unscaled)

X = Xorg[:-lead_time,:]

X = include_time_lag(X, max_lag=time_lag)
Xfull = include_time_lag(Xorg, max_lag=time_lag)

yorg = nino_to_category(nino34.values, threshold=threshold, categories=None)
yshift_predict = yorg[time_lag:-lead_time]
#%%
y = yorg[lead_time + time_lag:]
timey = nino34.index[lead_time + time_lag:]
nino34y = nino34[lead_time + time_lag:]


train_end = int(train_frac * X.shape[0])
trainX, testX = X[:train_end,:], X[train_end:,:]
trainy, testy= y[:train_end], y[train_end:]
traintimey, testtimey = timey[:train_end], timey[train_end:]

nino34trainy, nino34testy = nino34y[:train_end], nino34y[train_end:]

# =============================================================================
# # neural network
# =============================================================================
model = Sequential()

model.add(Dense(32, input_dim=X.shape[1],activation='relu', kernel_regularizer=regularizers.l2(0.005)))
model.add(Dropout(0.2))
model.add(Dense(16, input_dim=X.shape[1],activation='relu', kernel_regularizer=regularizers.l2(0.005)))
model.add(Dropout(0.2))
model.add(Dense(classes, activation='softmax'))

optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_acc',
                          min_delta=0.0,
                          patience=20,
                          verbose=0, mode='auto',
                          restore_best_weights=True)

history = model.fit(trainX, trainy, epochs=250, batch_size=20,verbose=0,
                    shuffle=True, callbacks=[es],
                    validation_data=(testX, testy))

predicty = model.predict(testX)

#%% =============================================================================
# # plot
# =============================================================================
plt.close("all")
from ninolearn.plot.nino_timeseries import nino_background
plt.figure(1)
plt.title(f"Lead time: {lead_time} month")
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
plt.title(f"Confusion Matrix, lead time: {lead_time} month")
plot_confMat(testy, yshift_predict[train_end:], class_labels)
plt.title(f"Confusion Matrix (persistance), lead time: {lead_time} month")
