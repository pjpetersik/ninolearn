"""
This code was inspired by the instructions found on
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural
-networks-python-keras/

TODO: make the NRMSE the loss
"""
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

from ninolearn.IO.read_post import (data_reader, csv_vars)

from .augment import window_warping

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
        self.features_df = pd.DataFrame()

        first = True

        for key in self.feature_keys:
            self.features_df[key] = self._read_wrapper(key)

            if first:
                self.features = self._prepare_feature(key,
                                                      self.features_df[key].values)
                first = False
            else:
                new_feature = self._prepare_feature(key,
                                                    self.features_df[key].values)

                self.features = np.concatenate((self.features, new_feature),
                                               axis=2)

        self.__trainX, self.__testX = self._create_feature_set(self.features)
        self.trainXorg, self.testXorg = self.__trainX.copy(), self.__testX.copy()

        if False:
            for _ in range(20):
                first = True
                for key in self.feature_keys:
                    self.features_df[key] = self._read_wrapper(key)
                    feature_values = self.features_df[key].values
                    feature_warped = window_warping(feature_values)

                    if first:
                        features = self._prepare_feature(key, feature_warped)
                        first = False
                    else:
                        new_feature = self._prepare_feature(key, feature_warped)

                        features = np.concatenate((features, new_feature),
                                                       axis=2)
                warped_trainX, warped_testX = self._create_feature_set(features)

                self.__trainX = np.concatenate((self.__trainX, warped_trainX), axis = 0)
                self.__testX = np.concatenate((self.__testX, warped_testX), axis = 0)


    def load_label(self, key):
        """
        loads the features corresponding to a key into the object and directly
        scales it.

        :type key: str
        :param key: the name of the label that is used as the key in the
        dictionary
        """
        self.label_df = self._read_wrapper(key)
        label_values = self.label_df.values
        self.label = self._prepare_label(label_values)
        self.n_samples = self.label.shape[0]

        self.__trainY, self.__testY = self._create_label_set(self.label)
        self.trainYorg, self.testYorg = self.__trainY.copy(), self.__testY.copy()

        if False:
            for _ in range(20):
                label_warped = window_warping(label_values)
                label = self._prepare_label(label_warped)
                warped_trainY, warped_testY = self._create_label_set(label)

                self.__trainY = np.concatenate((self.__trainY, warped_trainY), axis = 0)
                self.__testY = np.concatenate((self.__testY, warped_testY), axis = 0)


    def _read_wrapper(self, key):
        """
        gets the right data belonging to a key word from the location
        dictionary.

        :param key: the key for the data_pool_dict.
        """
        ll = self.data_pool_dict[key]

        return self._read_data(statistic=ll[0], metric=ll[1], variable=ll[2],
                               processed=ll[3], dataset=ll[4])

    def _read_data(self, statistic=None, metric=None, variable=None,
                   processed='anom', dataset=None):
        """
        This method ensures that the right reader from the IO package is used
        for the desired variable
        """
        if variable in csv_vars:
            df = self._reader.read_csv(variable, processed=processed)

        elif statistic is not None:
            df = self._reader.read_statistic(statistic, variable,
                                             processed=processed,
                                             dataset=dataset)

            df = df[metric]

        # TODO: do this more elegent
        self.time = df.index
        return df

    def _prepare_feature(self, key, feature):
        """
        squashes featuere and reshapes it to the shape needed for the LSTM
        """
        feature = feature.astype('float32')
        feature = feature.reshape(len(feature), 1)
        feature = self.feature_scalers[key].fit_transform(feature)
        feature = feature.reshape(len(feature), 1, 1)
        return feature

    def _prepare_label(self, label):
        """
        squashes label and reshapes it to the shape needed for the LSTM
        """
        label = label.astype('float32')
        label = label.reshape(len(label), 1)
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
            a = X[i:i + self.window_size, 0, :]
            dataX.append(a)
        return np.array(dataX)

    def _restructure_label(self, Y):
        dataY = []
        for i in range(len(Y) - self.window_size - self.lead_time + 1):
            dataY.append(Y[i + self.window_size + self.lead_time - 1, 0])
        return np.array(dataY)

    def _restructure_time(self, time):
        """
        returns the time stamps at which predictions start
        """
        dataTime = []
        for i in range(len(time) - self.window_size - self.lead_time + 1):
            dataTime.append(time[i + self.window_size - 1])
        return pd.to_datetime(dataTime)

    def _create_feature_set(self, features):
        trainX, testX = self._split(features)

        trainX = self._restructure_feature(trainX)
        testX = self._restructure_feature(testX)

        return trainX, testX

    def _create_label_set(self, label):
        trainY, testY = self._split(label)
        trainYtime, testYtime = self._split(self.label_df.index)

        trainY = self._restructure_label(trainY)
        self.trainYtime = self._restructure_time(trainYtime)

        testY = self._restructure_label(testY)
        self.testYtime = self._restructure_time(testYtime)
        return trainY, testY

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
    def __init__(self, DataInstance, Layers=LSTM, n_neurons=[10], Dropout=0.2,
                 lr=0.001, epochs=200, batch_size=20, es_epochs=20, verbose=1):
        """
        A class to build and fit a recurrent neural network as well as use the
        RNN for predictions.

        :param DataInstance: An instance of the Data class

        :type Layers: list
        :param Layers: A list of layers. The first layer must be a
        keras.layers.recurrent class either SimpleRNN, GRU or
        LSTM. After the last LSTM layer Dense layer can follow.

        :type n_neurons: list
        :param n_neurons: the number of neurons in a layer

        :type Dropout: float
        :param Droput: the fraction of neurons turned of in the Dropout layer

        :type lr: float
        :param lr: learning rate

        :type epochs: int
        :param epochs: the number of epochs

        :type batch_size: int
        :param batch_size: the batch_size for the traing

        :type es_epochs: int
        :param es_epochs: number of epochs for early stopping

        :param verbose: the verbose argument for the model compile
        """

        # clear memory
        K.clear_session()

        assert isinstance(DataInstance, Data)
        self.layer_names = []
        for i in range(len(Layers)):
            assert (Layers[i].__module__ == 'keras.layers.recurrent') or\
                   (Layers[i].__module__ == 'keras.layers.core')
            self.layer_names.append(Layers[i].__name__)
        assert type(n_neurons) is list

        self.Layers = Layers

        self.n_neurons = n_neurons
        self.n_layers = len(n_neurons)

        assert len(self.Layers) == len(self.n_neurons)

        self.Dropout = Dropout
        self.architecture = {'layers': self.n_layers,
                             'n_neurons': self.n_neurons,
                             'layer_type': self.layer_names}

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

        self.verbose = verbose

        # extract some data form the Data instance
        self.trainY = self.Data.trainY
        self.testY = self.Data.testY

        self.trainYtime = self.Data.trainYtime
        self.testYtime = self.Data.testYtime

    def build_model(self):
        """
        This methods builds the model based on the information provided during
        the initialization of the instance.
        """
        self.model = Sequential()
        self.n_dense = self.Layers.count(Dense)
        self.n_recurrent = self.n_layers - self.n_dense

        for i in range(self.n_recurrent):
            if i == 0 and self.n_recurrent > 1:
                self.model.add(self.Layers[i](self.n_neurons[i],
                               input_shape=(self.window_size, self.n_features),
                               return_sequences=True))

            elif i == (self.n_recurrent - 1):
                self.model.add(self.Layers[i](self.n_neurons[i],
                               input_shape=(self.window_size, self.n_features),
                               return_sequences=False))

            else:
                self.model.add(self.Layers[i](self.n_neurons[i],
                                              return_sequences=True))
            self.model.add(Dropout(self.Dropout))

        for j in np.arange(self.n_recurrent, self.n_layers):
            self.model.add(self.Layers[j](self.n_neurons[j], activation="relu"))

        self.model.add(Dense(1, activation="linear"))

    def fit(self):
        """
        Wrapper function for the .fit() method of the keras model
        """
        self.history = self.model.fit(self.Data.trainX, self.Data.trainY,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size,
                                      verbose=self.verbose,
                                      validation_data=(self.Data.testX,
                                                       self.Data.testY),
                                      callbacks=[self.es],
                                      shuffle=True)

    def predict(self):
        """
        Wrapper function of the .predict() method of the keras model
        """
        self.trainPredict = self.model.predict(self.Data.trainXorg)
        self.testPredict = self.model.predict(self.Data.testXorg)

        shiftPredict = np.roll(self.Data.label, self.Data.lead_time)
        self.shiftY = self.Data.label[self.Data.lead_time:]
        self.shiftPredict = shiftPredict[self.Data.lead_time:]

    def get_scores(self, part):
        """
        Get the RMSE and NRMSE scores of the prediction.

        :type part: str
        :param part: the part from the data for which the scores should be
        returned. Either "train", "test" or "shift" for score on trainings,
        or test data or score of prediction by simply shifting the data by the
        lead_time.

        :return: Returns the RMSE and NRMSE
        """
        if part == 'train':
            RMSE = self._rmse(self.Data.trainYorg, self.trainPredict[:, 0])
            NRMSE = self._nrmse(self.Data.trainYorg, self.trainPredict[:, 0])
        elif part == 'test':
            RMSE = self._rmse(self.Data.testYorg, self.testPredict[:, 0])
            NRMSE = self._nrmse(self.Data.testYorg, self.testPredict[:, 0])
        elif part == 'shift':
            RMSE = self._rmse(self.shiftY, self.shiftPredict)
            NRMSE = self._nrmse(self.shiftY, self.shiftPredict)
        return RMSE, NRMSE

    def _rmse(self, y, predict):
        """
        Computes the root mean square error (RMSE)

        :param y: the base line data
        :param predict: the predicted data
        :return: the RMSE
        """
        return math.sqrt(mean_squared_error(y, predict))

    def _nrmse(self, y, predict):
        """
        Computes the nromalized root mean square error (NRMSE)

        :param y: the base line data
        :param predict: the predicted data
        :return: the NRMSE
        """
        return self._rmse(y, predict) / (np.max([y, predict])
                                         - np.min([y, predict]))

    def plot_history(self):
        """
        Plot the history of the model (RMSE)
        """
        plt.subplots()
        plt.plot(np.sqrt(self.history.history['mean_squared_error']),
                 label="MSE train")
        plt.plot(np.sqrt(self.history.history['val_mean_squared_error']),
                 label="MSE test")
        plt.legend()

    def plot_prediction(self):
        """
        P
        """
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
        shiftPredictPlot[self.Data.lead_time:, :] = self.shiftPredict

        # plot baseline and predictions

        fig, ax = plt.subplots()
        ax.plot(self.Data.time, self.Data.label,
                label="Nino3.4", c='k')
        ax.plot(self.Data.time, trainPredictPlot,
                label="trainPrediction",
                c='limegreen')
        ax.plot(self.Data.time, testPredictPlot,
                label="testPrediction",
                c='red')

        ax.plot(self.Data.time, shiftPredictPlot,
                label="shiftPrediction",
                ls='--', c='grey', alpha=0.5)

        ax.set_ylabel("Nino3.4")
        title = 'NRMSE test: %.2f, train: %.2f, shift: %.2f'\
                % (self.get_scores('test')[1],
                   self.get_scores('train')[1],
                   self.get_scores('shift')[1])
        ax.set_title(title)

        plt.gcf().autofmt_xdate()
        myFmt = mdates.DateFormatter('%Y-%m')

        plt.gca().xaxis.set_major_formatter(myFmt)
        ax.legend()
