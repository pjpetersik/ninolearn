# -*- coding: utf-8 -*-
import keras as kr
import numpy as np
import matplotlib.pyplot as plt

def restructure_feature(X, window_size=1, lead_time = 1):
        dataX = []
        for i in range(len(X) - window_size - lead_time + 1):
            a = X[i:i + window_size, 0, :]
            dataX.append(a)
        return np.array(dataX)


def restructure_label(y, window_size=1, lead_time = 1):
    dataY = []
    for i in range(len(y) - window_size - lead_time + 1):
        dataY.append(y[i + window_size + lead_time - 1])

    return np.array(dataY)

ws = 100
Xini = np.zeros(1000)
Xini[::10] = 1
Xini[11::20] = 1

X=Xini.reshape((1000,1,1))
X = restructure_feature(X, window_size=ws)

y = np.zeros(1000)
y[5::10] = 1
y[5::20] = 2


#y=y.reshape((1000,1))
y=restructure_label(y,window_size=ws)

#%%
model = kr.Sequential()

model.add(kr.layers.LSTM(10, activation='sigmoid', input_shape=(X.shape[1], X.shape[2]),
                         kernel_initializer='random_uniform', bias_initializer='random_uniform'))

model.add(kr.layers.Dense(1, activation='linear',
                           kernel_initializer='random_uniform', bias_initializer='random_uniform'))

optimizer = kr.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False)

model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mse'])

es = kr.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0.0,
                              patience=10,
                              verbose=0, mode='auto')

history = model.fit(X, y, epochs=100, batch_size=100,verbose=2, shuffle=False,
                    validation_split=0.4 ,callbacks=[es])


plt.close("all")
y_prediction = model.predict(X)
plt.figure(2)

plt.plot(y,label="data")
plt.plot(y_prediction,label="prediction")
plt.legend()
