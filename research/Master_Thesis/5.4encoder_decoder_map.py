# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from mpl_toolkits.basemap import Basemap
import pandas as pd

import keras.backend as K

from sklearn.preprocessing import StandardScaler


from ninolearn.IO.read_post import data_reader
from ninolearn.learn.models.encoderDecoder import EncoderDecoder
from ninolearn.pathes import ed_model_dir
from ninolearn.utils import print_header
from ninolearn.private import plotdir

from os.path import join
plt.close("all")

#%%free memory from old sessions
K.clear_session()
decades_arr = [60, 70, 80, 90, 100, 110]
n_decades = len(decades_arr)
# =============================================================================
# Animation
# =============================================================================

#%% =============================================================================
# Data
# =============================================================================
reader = data_reader(startdate='1959-11', enddate='2017-12')
sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom').rolling(time=3).mean()[2:]
oni = reader.read_csv('oni')[2:]

# select
feature = sst.copy(deep=True)
label = sst.copy(deep=True)

# preprocess data
feature_unscaled = feature.values.reshape(feature.shape[0],-1)
label_unscaled = label.values.reshape(label.shape[0],-1)

scaler_f = StandardScaler()
Xorg = scaler_f.fit_transform(feature_unscaled)

scaler_l = StandardScaler()
yorg = scaler_l.fit_transform(label_unscaled)

Xall = np.nan_to_num(Xorg)
yall = np.nan_to_num(yorg)

shift = 3
lead = 6
print_header(f'Lead time: {lead} months')

y = yall[lead+shift:]
X = Xall[:-lead-shift]

timey = oni.index[lead+shift:]

y_nino = oni[lead+shift:]

pred_full_oni = np.array([])
true_oni = np.array([])
timeytrue = pd.DatetimeIndex([])

pred_da_full = xr.zeros_like(label[lead+shift:,:,:])

for j in range(n_decades):
    decade = decades_arr[j]

    print_header(f'Predict: {1902+decade}-01-01 till {1911+decade}-12-01')

    # get test data
    test_indeces = test_indeces = (timey>=f'{1902+decade}-01-01') & (timey<=f'{1911+decade}-12-01')
    testX, testy, testtimey = X[test_indeces,:], y[test_indeces,:], timey[test_indeces]
    testnino  = y_nino[test_indeces]

    # load model corresponding to the test data
    model = EncoderDecoder()
    dir_name = f'ed_ensemble_decade{decade}_lead{lead}'
    model.load(location=ed_model_dir, dir_name=dir_name)

    # make prediction
    pred, _ = model.predict(testX)

    # reshape data into an xarray data array (da)
    pred_da = xr.zeros_like(label[lead+shift:,:,:][test_indeces])
    pred_da.values = scaler_l.inverse_transform(pred).reshape((testX.shape[0], label.shape[1], label.shape[2]))

    # fill full forecast array
    timeytrue = timeytrue.append(testtimey)
    pred_da_full.values[test_indeces] = pred_da.values

    # calculate the ONI
    pred_oni = pred_da.loc[dict(lat=slice(-5, 5), lon=slice(190, 240))].mean(dim="lat").mean(dim='lon')

    # make the full time series
    pred_full_oni = np.append(pred_full_oni, pred_oni)
    true_oni = np.append(true_oni, testnino)



#%% =============================================================================
# #Plots
# =============================================================================
levels_sst = np.arange(-3, 3.25, 0.25)
lon2, lat2 = np.meshgrid(pred_da_full.lon, pred_da_full.lat)



plt.close("all")

fig, axs = plt.subplots(2, 1, figsize=(6,4))


date = '1999-01-01'

m = []

for i in [0,1]:
    m.append(Basemap(projection='merc',llcrnrlat=-30,urcrnrlat=30,\
                llcrnrlon=120,urcrnrlon=280,lat_ts=5,resolution='c',ax=axs[i]))

    x, y = m[i](lon2, lat2)

    m[i].drawparallels(np.arange(-90., 120., 15.), labels=[1,0,0,0], color='grey')
    m[i].drawmeridians(np.arange(0., 360., 30.), color='grey')
    m[i].drawmapboundary(fill_color='white')
    m[i].drawcoastlines()



axs[0].set_title('ND(1998)J(1999)', size=14)
cs_sst = m[0].contourf(x, y, pred_da_full.loc[date], cmap=plt.cm.seismic,levels=levels_sst, extend='both')
axs[0].text(9.2e5, 5.3e6, 'Forecast', weight='bold', size=12,
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 7})


cs_sst = m[1].contourf(x, y, label.loc[date], cmap=plt.cm.seismic,levels=levels_sst, extend='both')
axs[1].text(9.2e5, 5.3e6, 'Observation', weight='bold', size=12,
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 7})




# Color bar
fig.subplots_adjust(right=0.7)
cax= fig.add_axes([0.75, 0.15, 0.02, 0.7])


fig.colorbar(cs_sst, cax=cax,label="T [K]")

plt.savefig(join(plotdir, f'ed_forecast_lead{lead}_{date[:-3]}.pdf'))