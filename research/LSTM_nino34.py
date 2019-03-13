from ninolearn.io import read_data
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

nino34 = read_data.nino34_anom().values

shift = 6
val = 24

X = nino34[:-shift]
y = nino34[shift:]


X = X.reshape(X.shape[0],1,1)
y = y.reshape(y.shape[0],1)

X_train, X_val = X[:-val,:,:],X[-val:,:,:]
y_train, y_val = y[:-val,:],y[-val:,:]


model = Sequential()


model.add(LSTM(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False)

model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mse'])

history = model.fit(X_train, y_train, epochs=50, batch_size=10,verbose=2,shuffle=True, validation_split=0.1)

#%%
y_predict = model.predict(X_val)
plt.figure(1)
plt.plot(y_val,label="val")
plt.plot(y_predict,label="predict")
plt.legend(frameon=False)
# plot learning metric

plt.figure(2)
plt.plot(history.history['mean_squared_error'],label="MSE training")
plt.plot(history.history['val_mean_squared_error'],label="MSE validation")
plt.legend(frameon=False)