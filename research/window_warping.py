# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import LSTM, GRU, SimpleRNN, Dense

from scipy import interpolate

from ninolearn.learn.rnn import Data, RNNmodel
from ninolearn.plot.evaluation import (plot_explained_variance,
                                       plot_correlations)

pool = {'c2_air': ['network_metrics', 'fraction_clusters_size_2', 'air_daily',
                   'anom', 'NCEP'],
        'c3_air': ['network_metrics', 'fraction_clusters_size_3', 'air_daily',
                   'anom', 'NCEP'],
        'c5_air': ['network_metrics', 'fraction_clusters_size_5', 'air_daily',
                   'anom', 'NCEP'],
        'tau': ['network_metrics', 'global_transitivity', 'air_daily', 'anom',
                'NCEP'],
        'C': ['network_metrics', 'avelocal_transmissivity', 'air_daily',
              'anom', 'NCEP'],
        'S': ['network_metrics', 'fraction_giant_component', 'air_daily',
              'anom', 'NCEP'],
        'L': ['network_metrics', 'average_path_length', 'air_daily', 'anom',
              'NCEP'],
        'H': ['network_metrics', 'hamming_distance', 'air_daily', 'anom',
              'NCEP'],
        'Hstar': ['network_metrics', 'corrected_hamming_distance', 'air_daily',
                  'anom',
                  'NCEP'],
        'nino34': [None, None, 'nino34', 'anom', None],
        'wwv': [None, None, 'wwv', 'anom', None],
        'pca1': ['pca', 'pca1', 'air', 'anom', 'NCEP'],
        'pca2': ['pca', 'pca2', 'vwnd', 'anom', 'NCEP'],
        'pca3': ['pca', 'pca2', 'uwnd', 'anom', 'NCEP'],
        }

window_size = 12
lead_time = 6

data_obj = Data(label_name="nino34", data_pool_dict=pool,
                window_size=window_size, lead_time=lead_time,
                startdate='1980-01', train_frac=0.6)

data_obj.load_features(['nino34',
                        #'pca1', 'pca2', 'pca3',
                        #  'c3_air', 'c5_air' ,'c2_air',
                        #'S', 'H', 'tau', 'C', 'L'
                        ])

def window_warping(ts, window_size = [3,12] , strength=[0.3, 0.8], amount = 12):
    """
    window warping for data augmentation of time series.

    :param ts: the timeseries as a 1D np.ndarray

    :param window_size: Half the size of a window size for the warping.

    :param strength: the strength of the stretching/compressing. Float between
    0 and 1.

    :param amount: The amount how often a random window of the time series
    should be warped
    """
    assert type(ts) is np.ndarray

    len_ts = len(ts)
    x = np.arange(len_ts)
    f = interpolate.interp1d(x, ts)
    new_ts = np.zeros_like(ts)
    new_ts[:] = ts

    if type(window_size) == list:
        ws = np.random.randint(window_size[0], window_size[1])
    else:
        ws = window_size

    if type(strength) == list:
        s = np.random.uniform(strength[0], strength[1])
    else:
        s = strength


    for _ in range(amount):
        middle = np.random.randint(ws,len_ts-ws)
        begin = middle - ws
        end = middle + ws

        x_middle_shifted = x[middle] + ws * s * np.random.choice([-1,1])

        x1= np.linspace(x[begin], x_middle_shifted, ws, endpoint=False)
        x2 = np.linspace(x_middle_shifted, x[end], ws)

        new_ts[begin:middle] = f(x1)
        new_ts[middle:end] = f(x2)

    return new_ts



a = np.zeros_like(data_obj.label[:,0])
a[:] = data_obj.label[:,0]
a_new = window_warping(a)

plt.close("all")
plt.plot(a)
plt.plot(a_new)