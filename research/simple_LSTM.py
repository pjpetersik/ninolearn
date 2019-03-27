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
from keras.layers import LSTM
from keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from ninolearn.IO.read_post import data_reader


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


# normalize the dataset
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))

# TODO: better coding regaring the following lines
nino = nino.reshape(len(nino), 1)
nino = scaler1.fit_transform(nino)
dataset1 = nino.reshape(len(nino), 1, 1)

dataset2 = dataset2.reshape(len(dataset2), 1)
dataset2 = scaler2.fit_transform(dataset2)
dataset2 = dataset2.reshape(len(dataset2), 1, 1)

# just nino3.4 index
# dataset = dataset1

# nino3.4 and wwv
dataset = np.concatenate((dataset1, dataset2), axis=2)

# some structural information
n_timesteps = dataset.shape[0]
n_features = dataset.shape[2]


# split into train and test sets
train_size = int(n_timesteps * 0.67)
test_size = n_timesteps - train_size
train, test = dataset[:train_size, :, :], dataset[train_size:, :, :]

# reshape into X=t and Y=t+leadtime
windowsize = 24
lead_time = 6

trainX, trainY = create_dataset(train, windowsize, lead_time)
testX, testY = create_dataset(test, windowsize, lead_time)

# %%create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(windowsize, n_features)))
model.add(Dense(1))
optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False)
model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mse'])
model.fit(trainX, trainY, epochs=100, batch_size=10, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler1.inverse_transform(trainPredict)
trainY = scaler1.inverse_transform([trainY])
testPredict = scaler1.inverse_transform(testPredict)
testY = scaler1.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))


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
plt.close("all")
plt.plot(df.index.date, scaler1.inverse_transform(nino), label="Nino3.4")
plt.plot(df.index.date, trainPredictPlot, label="trainPrediction")
plt.plot(df.index.date, testPredictPlot, label="testPrediction")
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%Y-%m')
# ax.xaxis.set_major_formatter(myFmt)
plt.gca().xaxis.set_major_formatter(myFmt)
plt.legend()
plt.show()
