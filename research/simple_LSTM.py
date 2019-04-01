import matplotlib.pyplot as plt
import numpy as np

from keras.layers import LSTM, GRU, SimpleRNN, Dense

from ninolearn.learn.rnn import Data, RNNmodel


pool = {'c2_air': ['network_metrics', 'fraction_clusters_size_2', 'air_daily', 'anom',
                   'NCEP'],
        'c3_air': ['network_metrics', 'fraction_clusters_size_3', 'air_daily', 'anom',
                   'NCEP'],
        'c5_air': ['network_metrics', 'fraction_clusters_size_5', 'air_daily', 'anom',
                   'NCEP'],
        'tau': ['network_metrics', 'global_transitivity', 'air_daily', 'anom', 'NCEP'],
        'C': ['network_metrics', 'avelocal_transmissivity', 'air_daily', 'anom', 'NCEP'],
        'S': ['network_metrics', 'fraction_giant_component', 'air_daily', 'anom', 'NCEP'],
        'L': ['network_metrics', 'average_path_length', 'air_daily', 'anom', 'NCEP'],
        'H': ['network_metrics', 'hamming_distance', 'air_daily', 'anom', 'NCEP'],
        'Hstar': ['network_metrics', 'corrected_hamming_distance', 'air_daily', 'anom',
                  'NCEP'],
        'nino34': [None, None, 'nino34', 'anom', None],
        'wwv': [None, None, 'wwv', 'anom', None],
        'pca1': ['pca', 'pca1', 'air', 'anom', 'NCEP'],
        'pca2': ['pca', 'pca2', 'vwnd', 'anom', 'NCEP'],
        'pca3': ['pca', 'pca2', 'uwnd', 'anom', 'NCEP'],

        }

window_size = 6
lead_time = 3

data_obj = Data(label_name="nino34", data_pool_dict=pool,
                window_size=window_size, lead_time=lead_time,
                startdate='1980-01', train_frac=0.6)

data_obj.load_features(['wwv', # 'nino34',
                        'pca1', 'pca2', 'pca3',
                        'c2_air',  'c3_air', 'c5_air',
                        'S', 'H', 'tau', 'C', 'L'
                        ])

model = RNNmodel(data_obj, Layers=[LSTM], n_neurons=[10], Dropout=0.0,
                 lr=0.0001, epochs=5000, batch_size=2, es_epochs=20)

model.fit()
model.predict()

trainRMSE, trainNRMSE = model.get_scores('train')
testRMSE, testNRMSE = model.get_scores('test')
shiftRMSE, shiftNRMSE = model.get_scores('shift')

print('Train Score: %.2f MSE, %.2f NMSE' % (trainRMSE**2, trainNRMSE))
print('Test Score: %.2f MSE, %.2f NMSE' % (testRMSE**2, testNRMSE))
print('Shift Score: %.2f MSE, %.2f NMSE' % (shiftRMSE**2, shiftNRMSE))

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
    y = scale(model.testY[month])
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
ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J',
                    'J', 'A', 'S', 'O', 'N', 'D'])
