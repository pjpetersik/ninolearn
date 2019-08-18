# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from os.path import join

from keras import backend as K

from ninolearn.learn.models.dem import DEM
from ninolearn.pathes import modeldir, processeddir

from ninolearn.utils import print_header
from data_pipeline import pipeline

plt.close("all")

#%% =============================================================================
#  process data
# =============================================================================

pred_mean_save = np.zeros((672, 4))
pred_std_save = np.zeros((672, 4))

decades_arr = [60, 70, 80, 90, 100, 110]


n_decades = len(decades_arr)

lead_time_arr = np.array([0, 3, 6, 9])
n_lead = len(lead_time_arr)

for i in range(n_lead):
    lead_time = lead_time_arr[i]
    print_header(f'Lead time: {lead_time} months')

    X, y, timey, y_persistance = pipeline(lead_time, return_persistance=True)

    pred_mean_full = np.array([])
    pred_std_full = np.array([])
    pred_persistance_full = np.array([])
    ytrue = np.array([])
    timeytrue = pd.DatetimeIndex([])

    for j in range(n_decades):
        decade = decades_arr[j]

        # free some memory
        K.clear_session()

        # make predictions
        print(f'Predict: {1902+decade}-01-01 till {1911+decade}-12-01')

        ens_dir=f'ensemble_decade{decade}_lead{lead_time}'
        model = DEM()
        model.load(location=modeldir, dir_name=ens_dir)

        test_indeces = (timey>=f'{1902+decade}-01-01') & (timey<=f'{1911+decade}-12-01')

        testX, testy, testtimey = X[test_indeces,:], y[test_indeces], timey[test_indeces]

        pred_mean, pred_std = model.predict(testX)

        # make the full time series
        pred_mean_full = np.append(pred_mean_full, pred_mean)
        pred_std_full = np.append(pred_std_full, pred_std)
        ytrue = np.append(ytrue, testy)
        timeytrue = timeytrue.append(testtimey)

    pred_mean_save[:,i] = pred_mean_full
    pred_std_save[:,i] = pred_std_full
#%%
ds = xr.Dataset({'UU DE mean': (['target_season', 'lead'],  pred_mean_save),
                 'UU DE std': (['target_season', 'lead'],  pred_std_save)
                 },
                 coords={'target_season': timeytrue,
                         'lead': lead_time_arr
                         }
                )

ds.to_netcdf(join(processeddir, f'DE_forecasts.nc'))