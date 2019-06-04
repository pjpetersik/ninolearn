# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd

from keras import backend as K

from ninolearn.learn.dem import DEM
from ninolearn.pathes import modeldir
from ninolearn.learn.evaluation import rmse, correlation, rmse_mon
from ninolearn.plot.evaluation import plot_seasonal_skill
from ninolearn.utils import print_header
from data_pipeline import pipeline
from ninolearn.utils import scale

plt.close("all")
# starting year of el nino
elnino_ep = [1957, 1965, 1972, 1976, 1982,
             1997, 2016]

elnino_cp = [1953, 1958, 1963, 1968, 1969,
             1977, 1979, 1986, 1987, 1991,
             1994, 2002, 2004, 2006, 2009]

lanina_ep = [1964, 1970, 1973, 1988, 1998,
             2007, 2010]

lanina_cp  = [1954, 1955, 1967, 1971, 1974,
              1975, 1984, 1995, 2000, 2001, 2011]

#%% =============================================================================
#  process data
# =============================================================================
decades = [80, 90, 100, 110]
#decades = [100]

lead_time_arr = np.array([0, 3, 6, 9, 12, 15])

n_pred = len(lead_time_arr)
all_season_corr = np.zeros(n_pred)
all_season_rmse = np.zeros(n_pred)
all_season_corr_pres = np.zeros(n_pred)
all_season_rmse_pres = np.zeros(n_pred)
all_season_nll = np.zeros(n_pred)

seas_corr = np.zeros((12, n_pred))
seas_corr_pers = np.zeros((12, n_pred))
seas_rmse = np.zeros((12, n_pred))
seas_rmse_pers = np.zeros((12, n_pred))

for i in range(n_pred):
    lead_time = lead_time_arr[i]
    print_header(f'Lead time: {lead_time} months')

    X, y, timey, y_persistance = pipeline(lead_time, return_persistance=True)

    pred_mean_full = np.array([])
    pred_std_full = np.array([])
    pred_persistance_full = np.array([])
    ytrue = np.array([])
    timeytrue = pd.DatetimeIndex([])

    for decade in decades:
        K.clear_session()
        print(f'Predict: {1902+decade}-01-01 till {1911+decade}-12-01')

        ens_dir=f'ensemble_decade{decade}_lead{lead_time}'
        model = DEM()
        model.load(location=modeldir, dir_name=ens_dir)

        test_indeces = (timey>=f'{1902+decade}-01-01') & (timey<=f'{1911+decade}-12-01')
        testX, testy, testtimey = X[test_indeces,:], y[test_indeces], timey[test_indeces]

        pred_mean, pred_std = model.predict(testX)
        pred_pers = y_persistance[test_indeces]

        pred_mean_full = np.append(pred_mean_full, pred_mean)
        pred_std_full = np.append(pred_std_full, pred_std)
        pred_persistance_full = np.append(pred_persistance_full, pred_pers)
        ytrue = np.append(ytrue, testy)
        timeytrue = timeytrue.append(testtimey)

    # extract the years coresponding to a certain type
    type_indeces = np.zeros(len(timey), dtype=bool)

    for year in elnino_cp:
        type_indeces =  ((timey>=f'{year}-07-01') & (timey<=f'{year+1}-06-01')) | type_indeces

    #%% ===========================================================================
    # Deep ensemble
    # =============================================================================

    # all seasons skills
    all_season_corr[i] = np.corrcoef(ytrue[type_indeces], pred_mean_full[type_indeces])[0,1]
    all_season_corr_pres[i] = np.corrcoef(ytrue[type_indeces], pred_persistance_full[type_indeces])[0,1]

    ax = plt.figure().gca()
    plt.title(f'Lead time: {lead_time} month')
    plt.scatter(scale(ytrue[~type_indeces]), scale(pred_mean_full[~type_indeces]), c='k')
    plt.scatter(scale(ytrue[type_indeces]), scale(pred_mean_full[type_indeces]),c='r')

    plt.plot([-10,10],[-10,10], 'k')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.xlabel('Normalized Nino3.4')
    plt.ylabel('Normalized mean prediction')
#%%


ax = plt.figure().gca()
plt.plot(lead_time_arr, all_season_corr, label="Deep Ensemble Mean")
plt.plot(lead_time_arr, all_season_corr_pres, label="Persistence")
plt.ylim(-0.2,1)
plt.xlim(0,8)
plt.xlabel('lead time')
plt.ylabel('r')
plt.title('Correlation skill')
plt.grid()
plt.legend()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))


