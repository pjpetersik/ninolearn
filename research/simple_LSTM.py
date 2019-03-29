"""
This code follows the instructions found on https://machinelearningmastery.com/
time-series-prediction-lstm-recurrent-neural-networks-python-keras/
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import math

import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from ninolearn.IO.read_post import (data_reader, csv_vars, network_vars,
                                    netcdf_vars)


class Data(object):
    def __init__(self, label_name=None, data_pool_dict=None,
                 window_size=3, lead_time=3, train_frac=0.67,
                 startdate='1980-01', enddate='2018-12'):
        """
        :param data_pool_dict: A dictionary that links keywords, describing
        one particular variable to its meta information where it should be
        read. The item of a key word is a list of length 4 with entries
        [network_metric, variable, processed, dataset].
        """

        self.label_name = label_name

        self.window_size = window_size
        self.lead_time = lead_time

        self.train_frac = train_frac

        self.startdate = startdate
        self.enddate = enddate

        self._reader = data_reader(startdate=self.startdate,
                                   enddate=self.enddate)

        self.data_pool_dict = data_pool_dict

        self.load_label(self.label_name)

    def load_features(self, feature_keys):
        """
        loads the features corresponding to a key into the object and directly
        scales it.
        """
        assert type(feature_keys) is list
        self.feature_keys = feature_keys
        self.n_features = len(feature_keys)

        self.feature_scalers = self._set_scalers(self.feature_keys)
        first = True

        for key in self.feature_keys:
            if first:
                self.features = self._read_wrapper(key)
                self.features = self._prepare_feature(key, self.features)
                first = False
            else:
                new_feature = self._read_wrapper(key)
                new_feature = self._prepare_feature(key, new_feature)
                self.features = np.concatenate((self.features, new_feature),
                                               axis=2)
        self._create_feature_set()

    def load_label(self, key):
        """
        loads the features corresponding to a key into the object and directly
        scales it.
        """
        self.label_scalers = self._set_scalers(key)
        self.label = self._read_wrapper(key)
        self.label = self._prepare_label(key, self.label)
        self.n_samples = self.label.shape[0]
        self._create_label_set()

    def _read_wrapper(self, key):
        """
        gets the right data belonging to a key word from the location
        dictionary.

        :param key: the key for the data_pool_dict.
        """
        ll = self.data_pool_dict[key]

        return self._read_data(network_metric=ll[0], variable=ll[1],
                               processed=ll[2], dataset=ll[3])

    def _read_data(self, network_metric=None, variable=None, processed='anom',
                   dataset=None):
        """
        This method ensures that the right reader from the IO package is used
        for the desired variable
        """
        if variable in csv_vars:
            df = self._reader.read_csv(variable, processed=processed)

        elif variable in netcdf_vars and network_metric in network_vars:
            # returns network metrics belonging to a variable
            netdf = self._reader.read_network_metrics(variable,
                                                      processed=processed,
                                                      dataset=dataset)

            # get a specific network metric from the data set
            df = netdf[network_metric]
        # TODO: do this more elegent
        self.time_coord = df.index
        return df.values

    def _prepare_feature(self, key, feature):
        """
        squashes featuere and reshapes it to the shape needed for the LSTM
        """
        feature = feature.astype('float32')
        feature = feature.reshape(len(feature), 1)
        feature = self.feature_scalers[key].fit_transform(feature)
        feature = feature.reshape(len(feature), 1, 1)
        return feature

    def _prepare_label(self, key, label):
        """
        squashes label and reshapes it to the shape needed for the LSTM
        """
        label = label.astype('float32')
        label = label.reshape(len(label), 1)
        label = self.label_scalers[key].fit_transform(label)
        return label

    def _set_scalers(self, keys):
        scalers = {}
        if type(keys) is list:
            for key in keys:
                scalers[key] = MinMaxScaler(feature_range=(0, 1))
        elif type(keys) is str:
            scalers[keys] = MinMaxScaler(feature_range=(0, 1))
        return scalers

    def _split(self, data):
        self.train_size = int(self.n_samples * self.train_frac)
        self.test_size = self.n_samples - self.train_size

        train = data[:self.train_size]
        test = data[self.train_size:]
        return train, test

    def _restructure_feature(self, X):
        dataX = []
        for i in range(len(X) - self.window_size - self.lead_time + 1):
            a = X[i:(i + self.window_size), 0, :]
            dataX.append(a)
        return np.array(dataX)

    def _restructure_label(self, Y):
        dataY = []
        for i in range(len(Y) - self.window_size - self.lead_time + 1):
            dataY.append(Y[i + self.window_size + self.lead_time - 1, 0])
        return np.array(dataY)

    def _create_feature_set(self):
        trainX, testX = self._split(self.features)

        self.__trainX = self._restructure_feature(trainX)
        self.__testX = self._restructure_feature(testX)

    def _create_label_set(self):
        trainY, testY = self._split(self.label)
        self.__trainY = self._restructure_label(trainY)
        self.__testY = self._restructure_label(testY)

    @property
    def trainX(self):
        return self.__trainX

    @property
    def trainY(self):
        return self.__trainY

    @property
    def testX(self):
        return self.__testX

    @property
    def testY(self):
        return self.__testY


class RNNmodel(object):
    def __init__(self, DataInstance, n_layers=1, Layer=LSTM, n_neurons=[10],
                 lr=0.001, epochs=200, batch_size=20, es_epochs=20):
        """
        LSTMmodel to predict some time series
        """
        assert isinstance(DataInstance, Data)
        assert type(n_layers) is int
        assert Layer.__module__ == 'keras.layers.recurrent'
        assert type(n_neurons) is list
        assert len(n_neurons) == n_layers

        self.n_layers = n_layers
        self.Layer = Layer

        self.n_neurons = n_neurons

        self.architecture = {'layers': n_layers,
                             'n_neurons': n_neurons,
                             'layer_type': Layer.__name__}

        self.Data = DataInstance

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.window_size = self.Data.window_size
        self.n_features = self.Data.n_features

        self.build_model()

        self.optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None,
                              decay=0.0, amsgrad=False)

        self.model.compile(loss="mean_squared_error",
                           optimizer=self.optimizer,
                           metrics=['mse'])

        self.es = EarlyStopping(monitor='val_loss', min_delta=0.0,
                                patience=es_epochs, verbose=0, mode='auto')

    def build_model(self):
        self.model = Sequential()

        for i in range(self.n_layers):
            if i == 0 and self.n_layers > 1:
                self.model.add(self.Layer(self.n_neurons[i],
                               input_shape=(self.window_size, self.n_features),
                               return_sequences=True))

            elif i == 0 and self.n_layers == 1:
                self.model.add(self.Layer(self.n_neurons[i],
                               input_shape=(self.window_size, self.n_features),
                               return_sequences=False))

            else:
                self.model.add(self.Layer(self.n_neurons[i],
                                          return_sequences=False))
            self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

    def fit(self):
        self.history = self.model.fit(self.Data.trainX, self.Data.trainY,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size, verbose=2,
                                      validation_data=(self.Data.testX,
                                                       self.Data.testY),
                                      callbacks=[self.es])

    def predict(self):
        trainY = self.Data.trainY
        testY = self.Data.testY

        trainPredict = self.model.predict(self.Data.trainX)
        testPredict = self.model.predict(self.Data.testX)
        shiftPredict = np.roll(self.Data.label, self.Data.lead_time)

        self.label_scaler = self._get_label_scaler()

        # invert predictions
        self.trainPredict = self.label_scaler.inverse_transform(trainPredict)
        self.trainY = self.label_scaler.inverse_transform([trainY])

        self.testPredict = self.label_scaler.inverse_transform(testPredict)
        self.testY = self.label_scaler.inverse_transform([testY])
        self.shiftY = self.label_scaler.inverse_transform(
                     self.Data.label[self.Data.lead_time:])
        self.shiftPredict = self.label_scaler.inverse_transform(
               shiftPredict[self.Data.lead_time:])

    def get_scores(self, part):
        if part == 'train':
            RMSE = self._rmse(self.trainY[0], self.trainPredict[:, 0])
            NRMSE = self._nrmse(self.trainY[0], self.trainPredict[:, 0])
        elif part == 'test':
            RMSE = self._rmse(self.testY[0], self.testPredict[:, 0])
            NRMSE = self._nrmse(self.testY[0], self.testPredict[:, 0])
        elif part == 'shift':
            RMSE = self._rmse(self.shiftY, self.shiftPredict)
            NRMSE = self._nrmse(self.shiftY, self.shiftPredict)
        return RMSE, NRMSE

    def _rmse(self, y, predict):
        return math.sqrt(mean_squared_error(y, predict))

    def _nrmse(self, y, predict):
        return self._rmse(y, predict) / (np.max([y, predict])
                                         - np.min([y, predict]))

    def _get_label_scaler(self):
        return self.Data.label_scalers[self.Data.label_name]

    def plot_history(self):
        plt.subplots()
        plt.plot(self.history.history['mean_squared_error'],
                 label="MSE train")
        plt.plot(self.history.history['val_mean_squared_error'],
                 label="MSE test")
        plt.legend()

    def plot_prediction(self):
        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(self.Data.label)
        trainPredictPlot[:, :] = np.nan

        # begin and end indeces
        ibegtrain = self.window_size + self.Data.lead_time - 1
        iendtrain = ibegtrain + len(self.trainPredict)
        trainPredictPlot[ibegtrain:iendtrain, :] = self.trainPredict

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(self.Data.label)
        testPredictPlot[:, :] = np.nan

        # begin index
        ibegtest = iendtrain + self.window_size + self.Data.lead_time - 1
        testPredictPlot[ibegtest:, :] = self.testPredict

        shiftPredictPlot = np.empty_like(self.Data.label)
        shiftPredictPlot[:, :] = np.nan
        shiftPredictPlot[lead_time:, :] = self.shiftPredict

        # plot baseline and predictions

        plt.subplots()
        plt.plot(self.Data.time_coord,
                 self.label_scaler.inverse_transform(self.Data.label),
                 label="Nino3.4", c='k')
        plt.plot(self.Data.time_coord, trainPredictPlot,
                 label="trainPrediction",
                 c='limegreen')
        plt.plot(self.Data.time_coord, testPredictPlot,
                 label="testPrediction",
                 c='red')

        plt.plot(self.Data.time_coord, shiftPredictPlot,
                 label="shiftPrediction",
                 ls='--', c='grey', alpha=0.5)

        plt.gcf().autofmt_xdate()
        myFmt = mdates.DateFormatter('%Y-%m')
        # ax.xaxis.set_major_formatter(myFmt)
        plt.gca().xaxis.set_major_formatter(myFmt)
        plt.legend()
        plt.show()


pool = {'c2_air': ['fraction_clusters_size_2', 'air_daily', 'anom',
                   'NCEP'],
        'c3_air': ['fraction_clusters_size_3', 'air_daily', 'anom',
                   'NCEP'],
        'c5_air': ['fraction_clusters_size_5', 'air_daily', 'anom',
                   'NCEP'],
        'tau': ['global_transitivity', 'air_daily', 'anom', 'NCEP'],
        'C': ['avelocal_transmissivity', 'air_daily', 'anom', 'NCEP'],
        'S': ['fraction_giant_component', 'air_daily', 'anom', 'NCEP'],
        'L': ['average_path_length', 'air_daily', 'anom', 'NCEP'],
        'H': ['hamming_distance', 'air_daily', 'anom', 'NCEP'],
        'Hstar': ['corrected_hamming_distance', 'air_daily', 'anom',
                  'NCEP'],
        'nino34': [None, 'nino34', 'anom', None],
        'wwv': [None, 'wwv', 'anom', None]}

window_size = 24
lead_time = 12

data_obj = Data(label_name="nino34", data_pool_dict=pool,
                window_size=window_size, lead_time=lead_time,
                startdate='1950-01')

data_obj.load_features(['nino34', 'c2_air', 'c3_air', 'c5_air', 'S'])

model = RNNmodel(data_obj, n_layers=2, Layer=LSTM, n_neurons=[10, 10],
                 epochs=2)
model.fit()
model.predict()
trainRMSE, trainNRMSE = model.get_scores('train')
testRMSE, testNRMSE = model.get_scores('test')
shiftRMSE, shiftNRMSE = model.get_scores('shift')

print('Train Score: %.2f RMSE, %.2f NMSE' % (trainRMSE, trainNRMSE))
print('Test Score: %.2f RMSE, %.2f NMSE' % (testRMSE, testNRMSE))
print('Shift Score: %.2f RMSE, %.2f NMSE' % (shiftRMSE, shiftNRMSE))


plt.close("all")
model.plot_history()
model.plot_prediction()
