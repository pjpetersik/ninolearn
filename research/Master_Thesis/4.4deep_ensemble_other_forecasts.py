from os.path import join
import pandas as pd
import xarray as xr
import numpy as np
from ninolearn.pathes import rawdir, postdir
import matplotlib.pyplot as plt
from ninolearn.IO.read_post import data_reader
from ninolearn.plot.prediction import plot_prediction
plt.close('all')

start = '2003-01'
end = '2017-12'
reader = data_reader(startdate=start, enddate=end)
oni = reader.read_csv('oni')

data = xr.open_dataset(join(postdir, f'DE_forecasts.nc'))
data_of = xr.open_dataset(join(postdir, f'other_forecasts.nc'))

lead = 3
lead_DE = 1

UU_DE_mean = data['UU DE mean'].loc[start:end][:,lead_DE]
UU_DE_std = data['UU DE std'].loc[start:end][:,lead_DE]

plt.subplots(figsize=(9,3))
plot_prediction(UU_DE_mean.target_season.values, UU_DE_mean.values,std=UU_DE_std.values)

target_season = data_of['target_season'].loc[start:end].to_pandas()

UBC_NNET = data_of['UBC NNET'].loc[start:end, lead]
ECMWF = data_of['ECMWF'].loc[start:end, lead]
CFSv2 = data_of['NCEP CFSv2'].loc[start:end, lead]
UCLA_TCD = data_of['UCLA-TCD'].loc[start:end, lead]
JMA = data_of['JMA'].loc[start:end, lead]
NASA_GAMO = data_of['NASA GAMO'].loc[start:end, lead]

alpha = 0.8
plt.plot(target_season, UBC_NNET, alpha=alpha, label='UBC NNET')
plt.plot(target_season, ECMWF, alpha=alpha, label='ECMWF')
plt.plot(target_season, CFSv2, alpha=alpha, label='NCEP CFSv2')
plt.plot(target_season, UCLA_TCD, alpha=alpha, label='UCLA-TCD')
plt.plot(target_season, JMA, alpha=alpha, label='JMA')
plt.plot(target_season, NASA_GAMO, alpha=alpha, label='NASA GAMO')

plt.plot(oni.index, oni, c='k', label='ONI')

plt.xlim(UU_DE_mean.target_season.values[0], UU_DE_mean.target_season.values[-1])

plt.legend()



