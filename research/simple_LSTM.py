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

from ninolearn.IO.read_post import data_reader


class data(object):
    def __init__(self, ml_type, names):
        self.ml_type = ml_type
        self.names = names

    @property
    def values(self):
        return self._values

    @values.setter
    def assign_data(self, data):
        self._values = data


class LSTMmodel(object):
    def __init__(self, label_list, feature_list):
        """
        LSTMmodel to predict some time series
        """
        msg = "label and feature must be str type"
        assert type(label_list) is list
        msg = "label and feature list of str"
        assert type(feature_list) is list, msg
        # TODO: How to check for correct types?

        self.labels = data("label", label_list)
        self.features = data("features", feature_list)

    def get_data(self, startdate='1980-01', enddate='2018-12'):
        """
        Get data from database in postdir
        """
        self.data_startdate = startdate
        self.data_enddate = enddate

        for name in self.labels.names:
            label_data_reader = self._get_reader(name)
            self.labels.assign_data = label_data_reader()

    def _get_reader(self, variable_name):
        reader = data_reader(startdate=self.data_startdate,
                             enddate=self.data_enddate)

        reader_dict = {'nino': reader.nino34_anom,
                       'wwv': reader.wwv_anom}

        return reader_dict[variable_name]


if __name__ == "__main__":
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, windowsize=3, lead_time=3):
        dataX, dataY = [], []
        for i in range(len(dataset) - windowsize - lead_time + 1):
            a = dataset[i:(i + windowsize), 0, :]
            dataX.append(a)
            dataY.append(dataset[i + windowsize + lead_time - 1, 0, 0])
        return np.array(dataX), np.array(dataY)

    # fix random seed for reproducibility
    np.random.seed(7)

    # read data1
    reader = data_reader(startdate='1980-01')
    df = reader.nino34_anom()

    nino = df.values
    nino = nino.astype('float32')

    # read data2
    df2 = reader.wwv_anom()
    dataset2 = df2.values
    dataset2 = dataset2.astype('float32')

    # read data2
    df2 = reader.wwv_anom()
    dataset2 = df2.values
    dataset2 = dataset2.astype('float32')

    df3 = reader.read_network_metrics('air_daily',
                                      dataset='NCEP',
                                      processed='anom')

    dataset3 = df3['fraction_clusters_size_2'].values
    dataset3 = dataset3.astype('float32')

    dataset4 = df3['global_transitivity'].values
    dataset4 = dataset4.astype('float32')

    # normalize the dataset
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaler2 = MinMaxScaler(feature_range=(0, 1))
    scaler3 = MinMaxScaler(feature_range=(0, 1))
    scaler4 = MinMaxScaler(feature_range=(0, 1))

    # TODO: better coding regaring the following lines
    nino = nino.reshape(len(nino), 1)
    nino = scaler1.fit_transform(nino)
    dataset1 = nino.reshape(len(nino), 1, 1)

    dataset2 = dataset2.reshape(len(dataset2), 1)
    dataset2 = scaler2.fit_transform(dataset2)
    dataset2 = dataset2.reshape(len(dataset2), 1, 1)

    dataset3 = dataset3.reshape(len(dataset3), 1)
    dataset3 = scaler3.fit_transform(dataset3)
    dataset3 = dataset3.reshape(len(dataset3), 1, 1)

    dataset4 = dataset4.reshape(len(dataset4), 1)
    dataset4 = scaler4.fit_transform(dataset4)
    dataset4 = dataset4.reshape(len(dataset4), 1, 1)

    # nino3.4 and wwv
    dataset = np.concatenate((dataset1, dataset2, dataset3, dataset4), axis=2)

    # some structural information
    n_timesteps = dataset.shape[0]
    n_features = dataset.shape[2]

    # split into train and test sets
    train_size = int(n_timesteps * 0.67)
    test_size = n_timesteps - train_size
    train, test = dataset[:train_size, :, :], dataset[train_size:, :, :]

    # reshape into X=t and Y=t+leadtime
    windowsize = 48
    lead_time = 6

    trainX, trainY = create_dataset(train, windowsize, lead_time)
    testX, testY = create_dataset(test, windowsize, lead_time)

    # %%create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(8, input_shape=(windowsize, n_features),
                   return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                     epsilon=None, decay=0.0001, amsgrad=False)

    model.compile(loss="mean_squared_error",
                  optimizer=optimizer,
                  metrics=['mse'])

    es = EarlyStopping(monitor='val_loss',
                       min_delta=0.0,
                       patience=20,
                       verbose=0, mode='auto')

    history = model.fit(trainX, trainY,
                        epochs=1000, batch_size=20, verbose=2,
                        validation_data=(testX, testY), callbacks=[es])

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler1.inverse_transform(trainPredict)
    trainY = scaler1.inverse_transform([trainY])
    testPredict = scaler1.inverse_transform(testPredict)
    testY = scaler1.inverse_transform([testY])

    # %% calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # calculate normalized root mean squared error
    trainScore = trainScore / (np.max([trainY[0], trainPredict[:, 0]])
                               - np.min([trainY[0], trainPredict[:, 0]]))
    print('Train Score: %.2f NRMSE' % (trainScore))

    testScore = testScore / (np.max([testY[0], testPredict[:, 0]])
                             - np.min([testY[0], testPredict[:, 0]]))
    print('Test Score: %.2f NRMSE' % (testScore))

    # %%
    plt.close("all")

    plt.subplots()
    plt.plot(history.history['mean_squared_error'], label="MSE train")
    plt.plot(history.history['val_mean_squared_error'], label="MSE test")

    # %%shift train predictions for plotting
    trainPredictPlot = np.empty_like(nino)
    trainPredictPlot[:, :] = np.nan

    # begin and end indeces
    ibegtrain = windowsize + lead_time - 1
    iendtrain = ibegtrain + len(trainPredict)
    trainPredictPlot[ibegtrain:iendtrain, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(nino)
    testPredictPlot[:, :] = np.nan

    # begin index
    ibegtest = iendtrain + windowsize + lead_time - 1
    testPredictPlot[ibegtest:, :] = testPredict

    # plot baseline and predictions

    plt.subplots()
    plt.plot(df.index.date, scaler1.inverse_transform(nino), label="Nino3.4")
    plt.plot(df.index.date, trainPredictPlot, label="trainPrediction")
    plt.plot(df.index.date, testPredictPlot, label="testPrediction")
    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%Y-%m')
    # ax.xaxis.set_major_formatter(myFmt)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.legend()
    plt.show()
