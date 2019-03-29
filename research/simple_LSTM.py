import matplotlib.pyplot as plt
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
lead_time = 12

data_obj = Data(label_name="nino34", data_pool_dict=pool,
                window_size=window_size, lead_time=lead_time,
                startdate='1980-01')

data_obj.load_features(['nino34', 'wwv', 'c2_air', 'c3_air', 'c5_air',
                        'S', 'H', 'tau', 'C', 'L'])

model = RNNmodel(data_obj, Layer=LSTM, n_neurons=[10], Dropout=0.3, lr=0.001,
                 epochs=500, batch_size=50, es_epochs=10)

model.fit()
model.predict()
trainRMSE, trainNRMSE = model.get_scores('train')
testRMSE, testNRMSE = model.get_scores('test')
shiftRMSE, shiftNRMSE = model.get_scores('shift')

print('Train Score: %.2f RMSE, %.2f NMSE' % (trainRMSE, trainNRMSE))
print('Test Score: %.2f RMSE, %.2f NMSE' % (testRMSE, testNRMSE))
print('Shift Score: %.2f RMSE, %.2f NMSE' % (shiftRMSE, shiftNRMSE))
#%%
plt.close("all")
model.plot_history()
model.plot_prediction()
