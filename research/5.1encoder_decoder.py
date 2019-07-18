# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.animation as animation

import keras.backend as K

from sklearn.preprocessing import StandardScaler

from ninolearn.IO.read_post import data_reader
from ninolearn.learn.encoderDecoder import EncoderDecoder
from ninolearn.pathes import ed_model_dir
from ninolearn.utils import print_header

import gc
import os
import time

#%%free memory from old sessions
K.clear_session()
# =============================================================================
# Data
# =============================================================================
#read data
reader = data_reader(startdate='1959-11', enddate='2017-12')
sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom').rolling(time=3).mean()[2:]
taux = reader.read_netcdf('taux', dataset='NCEP', processed='anom').rolling(time=3).mean()[2:]

nino34 = reader.read_csv('nino3.4S')[2:]
#%%
# select
feature = sst.copy(deep=True)
feature2 = taux.copy(deep=True)
label = sst.copy(deep=True)

# preprocess data
feature_unscaled = feature.values.reshape(feature.shape[0],-1)
#feature2_unscaled = feature2.values.reshape(feature.shape[0],-1)

#feature_unscaled = np.concatenate((feature_unscaled, feature2_unscaled), axis=1)
label_unscaled = label.values.reshape(label.shape[0],-1)


scaler_f = StandardScaler()
Xorg = scaler_f.fit_transform(feature_unscaled)

scaler_l = StandardScaler()
yorg = scaler_l.fit_transform(label_unscaled)

Xall = np.nan_to_num(Xorg)
yall = np.nan_to_num(yorg)

# shift
shift = 3
for lead in [3, 6, 9, 12, 15, 0]:
    print_header(f'Lead time: {lead} month')

    y = yall[lead+shift:]
    X = Xall[:-lead-shift]
    timey = nino34.index[lead+shift:]

    for decade in [60, 70, 80, 90, 100, 110]:
        print_header(f'Test period: {1902+decade}-01-01 till {1911+decade}-12-01')
        K.clear_session()

        # jump loop iteration if already trained
        ens_dir=f'ed_ensemble_decade{decade}_lead{lead}'
        out_dir = os.path.join(ed_model_dir, ens_dir)

        modified_time = time.gmtime(os.path.getmtime(out_dir))
        compare_time = time.strptime("15-7-2019 13:00 UTC", "%d-%m-%Y %H:%M %Z")

        if modified_time>compare_time:
            print("Trained already!")
            continue

        test_indeces = test_indeces = (timey>=f'{1902+decade}-01-01') & (timey<=f'{1911+decade}-12-01')
        train_indeces = np.invert(test_indeces)

        trainX, trainy, traintimey = X[train_indeces,:], y[train_indeces,:], timey[train_indeces]
        testX, testy, testtimey = X[test_indeces,:], y[test_indeces,:], timey[test_indeces]

        # =============================================================================
        # Model
        # =============================================================================

        model = EncoderDecoder()

        model.set_parameters(neurons=[(32, 8), (512, 64)], dropout=[0., 0.2], noise=[0., 0.5] , noise_out=[0., 0.5],
                 l1_hidden=[0.0, 0.001], l2_hidden=[0.0, 0.001], l1_out=[0.0, 0.001], l2_out=[0.0, 0.001], batch_size=100,
                 lr=[0.0001, 0.01], n_segments=5, n_members_segment=1, patience = 40, epochs=1000, verbose=0)

        #model.fit(trainX, trainy)
        model.fit_RandomizedSearch(trainX, trainy, n_iter=50)
        dir_name = f'ed_ensemble_decade{decade}_lead{lead}'
        model.save(location=ed_model_dir, dir_name=dir_name)

        del model
        gc.collect()