"""
This code follows the instructions found on https://machinelearningmastery.com/
time-series-prediction-lstm-recurrent-neural-networks-python-keras/
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from ninolearn.IO.read_post import (data_reader, csv_vars, network_vars,
                                    netcdf_vars)


class LSTMmodel(object):
    def __init__(self, label_name, window_size=3, lead_time=3,
                 startdate='1980-01', enddate='2018-12'):
        """
        LSTMmodel to predict some time series
        """
        pass


class Data(object):
    def __init__(self, label_name="nino34", data_pool_dict=None,
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
        #TODO: do this more elegent
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

if __name__ == "__main__":

    #%% Generate the Data object

    pool = {'c2_air': ['fraction_clusters_size_2', 'air_daily', 'anom', 'NCEP'],
        'c3_air': ['fraction_clusters_size_3', 'air_daily', 'anom', 'NCEP'],
        'c5_air': ['fraction_clusters_size_5', 'air_daily', 'anom', 'NCEP'],
        'tau': ['global_transitivity', 'air_daily', 'anom', 'NCEP'],
        'C': ['avelocal_transmissivity', 'air_daily', 'anom', 'NCEP'],
        'S': ['fraction_giant_component', 'air_daily', 'anom', 'NCEP'],
        'L': ['average_path_length', 'air_daily', 'anom', 'NCEP'],
        'H': ['hamming_distance', 'air_daily', 'anom', 'NCEP'],
        'Hstar': ['corrected_hamming_distance', 'air_daily', 'anom', 'NCEP'],
        'nino34': [None, 'nino34', 'anom', None],
        'wwv': [None, 'wwv', 'anom', None]}

    window_size  = 24
    lead_time = 12
    data_obj = Data(data_pool_dict=pool, window_size=window_size,
                    lead_time=lead_time, startdate='1950-01')

    data_obj.load_features(['nino34', 'c2_air', 'c3_air', 'c5_air', 'S'])#,'C',


    # Extract some data object parameters
    trainX, trainY = data_obj.trainX, data_obj.trainY
    testX, testY = data_obj.testX, data_obj.testY

    n_features = data_obj.n_features
    scaler_label = data_obj.label_scalers['nino34']
    label = data_obj.label
    time = data_obj.time_coord

    # %%create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(32, input_shape=(window_size, n_features),
                   return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                     epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss="mean_squared_error",
                  optimizer=optimizer,
                  metrics=['mse'])

    es = EarlyStopping(monitor='val_loss',
                       min_delta=0.0,
                       patience=20,
                       verbose=0, mode='auto')

    history = model.fit(trainX, trainY,
                        epochs=1000, batch_size=10, verbose=2,
                        validation_data=(testX, testY), callbacks=[es])

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler_label.inverse_transform(trainPredict)
    trainY = scaler_label.inverse_transform([trainY])
    testPredict = scaler_label.inverse_transform(testPredict)
    testY = scaler_label.inverse_transform([testY])

    # %% calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    # calculate normalized root mean squared error
    trainScore = trainScore / (np.max([trainY[0], trainPredict[:, 0]])
                               - np.min([trainY[0], trainPredict[:, 0]]))
    print('Train Score: %.2f NRMSE' % (trainScore))

    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    testScore = testScore / (np.max([testY[0], testPredict[:, 0]])
                             - np.min([testY[0], testPredict[:, 0]]))
    print('Test Score: %.2f NRMSE' % (testScore))

    shiftY = scaler_label.inverse_transform(label[lead_time:])
    shiftPredict = np.roll(label, lead_time)
    shiftPredict = scaler_label.inverse_transform(shiftPredict[lead_time:])

    shiftScore = math.sqrt(mean_squared_error(shiftY, shiftPredict))
    print('Shift Score: %.2f RMSE' % (shiftScore))
    shiftScore = shiftScore / (np.max([shiftY, shiftPredict])
                               - np.min([shiftY, shiftPredict]))
    print('Shift Score: %.2f NRMSE' % (shiftScore))

    # %%
    plt.close("all")

    plt.subplots()
    plt.plot(history.history['mean_squared_error'], label="MSE train")
    plt.plot(history.history['val_mean_squared_error'], label="MSE test")

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(label)
    trainPredictPlot[:, :] = np.nan

    # begin and end indeces
    ibegtrain = window_size + lead_time - 1
    iendtrain = ibegtrain + len(trainPredict)
    trainPredictPlot[ibegtrain:iendtrain, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(label)
    testPredictPlot[:, :] = np.nan

    # begin index
    ibegtest = iendtrain + window_size + lead_time - 1
    testPredictPlot[ibegtest:, :] = testPredict

    shiftPredictPlot = np.empty_like(label)
    shiftPredictPlot[:, :] = np.nan
    shiftPredictPlot[lead_time:, :] = shiftPredict

    # plot baseline and predictions

    plt.subplots()
    plt.plot(time, scaler_label.inverse_transform(label),
             label="Nino3.4", c='k')
    plt.plot(time, trainPredictPlot, label="trainPrediction",
             c='limegreen')
    plt.plot(time, testPredictPlot, label="testPrediction",
             c='red')

    plt.plot(time, shiftPredictPlot, label="shiftPrediction",
             ls='--', c='grey', alpha=0.5)

    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%Y-%m')
    # ax.xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.legend()
    plt.show()
