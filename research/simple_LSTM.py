import matplotlib.pyplot as plt
import numpy as np

from keras.layers import LSTM, GRU, SimpleRNN

from ninolearn.learn.rnn import Data, RNNmodel


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
lead_time = 6

data_obj = Data(label_name="nino34", data_pool_dict=pool,
                window_size=window_size, lead_time=lead_time,
                startdate='1980-01')

data_obj.load_features(['nino34', 'wwv', 'c2_air', 'c3_air', 'c5_air', 'S', 'H', 'tau', 'C', 'L'])

model = RNNmodel(data_obj, Layers=[LSTM], n_neurons=[30], Dropout=0.5,
                 lr=0.00001, epochs=5, batch_size=100, es_epochs=5)

model.fit()
model.predict()

trainRMSE, trainNRMSE = model.get_scores('train')
testRMSE, testNRMSE = model.get_scores('test')
shiftRMSE, shiftNRMSE = model.get_scores('shift')

print('Train Score: %.2f RMSE, %.2f NMSE' % (trainRMSE, trainNRMSE))
print('Test Score: %.2f RMSE, %.2f NMSE' % (testRMSE, testNRMSE))
print('Shift Score: %.2f RMSE, %.2f NMSE' % (shiftRMSE, shiftNRMSE))

# %%

plt.close("all")
model.plot_history()
model.plot_prediction()


def scale(x):
    return (x-x.mean())/x.std()


# scatter
fig, ax = plt.subplots(4, 3, figsize=(8, 6))
plt.subplots_adjust(hspace=0.5)
pos = np.argwhere(np.zeros((4, 3)) == 0)

r = np.zeros(12)
rsq = np.zeros(12)
for i in range(0, 12):
    month = (model.Data.testYtime.month == i+1)
    y = scale(model.testY[0, month])
    pred = scale(model.testPredict[month, 0])
    r[i] = np.corrcoef(y, pred)[0, 1]
    rsq[i] = round(r[i]**2, 3)
    ax[pos[i, 0], pos[i, 1]].scatter(y, pred)
    ax[pos[i, 0], pos[i, 1]].set_xlim([-3, 3])
    ax[pos[i, 0], pos[i, 1]].set_ylim([-3, 3])

    ax[pos[i, 0], pos[i, 1]].set_title(f"month: {i}, r$^2$:{rsq[i]}")

# bar
m = np.arange(1, 13)
fig, ax = plt.subplots()
ax.set_ylim(0, 1)
ax.bar(m, rsq)
ax.set_xticks(m)
ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D',])
