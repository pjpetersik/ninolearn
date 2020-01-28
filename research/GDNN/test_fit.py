mport numpy as np
from sklearn.preprocessing import StandardScaler

from ninolearn.utils import include_time_lag
from ninolearn.IO.read_processed import data_reader
from ninolearn.learn.models.dem import DEM

reader = data_reader(startdate='1960-01', enddate='2017-12')

 # indeces
oni = reader.read_csv('oni')

iod = reader.read_csv('iod')
wwv = reader.read_csv('wwv_proxy')

# network metrics
network_ssh = reader.read_statistic('network_metrics', variable='zos', dataset='ORAS4', processed="anom")
c2_ssh = network_ssh['fraction_clusters_size_2']
H_ssh = network_ssh['corrected_hamming_distance']

#wind stress
taux = reader.read_netcdf('taux', dataset='NCEP', processed='anom')

taux_WP = taux.loc[dict(lat=slice(2.5,-2.5), lon=slice(120, 160))]
taux_WP_mean = taux_WP.mean(dim='lat').mean(dim='lon')

# decadel variation of leading eof
pca_dec = reader.read_statistic('pca', variable='dec_sst', dataset='ERSSTv5', processed='anom')['pca1']

lead_time = 3

# time lag
time_lag = 2

# shift such that lead time corresponds to the definition of lead time
shift = 3

# process features
feature_unscaled = np.stack((oni,
                             wwv,
                             iod,
                             taux_WP_mean,
                             c2_ssh,
                             H_ssh,
                             pca_dec
                             ), axis=1)

# scale each feature
scalerX = StandardScaler()
Xorg = scalerX.fit_transform(feature_unscaled)

# set nans to 0.
Xorg = np.nan_to_num(Xorg)

# arange the feature array
X = Xorg[:-lead_time-shift,:]
X = include_time_lag(X, max_lag=time_lag)

# arange label
yorg = oni.values
y = yorg[lead_time + time_lag + shift:]

# get the time axis of the label
timey = oni.index[lead_time + time_lag + shift:]

test_indeces = (timey>=f'2001-01-01') & (timey<=f'2011-12-01')
train_indeces = np.invert(test_indeces)
trainX, trainy = X[train_indeces,:], y[train_indeces]
testX, testy =  X[test_indeces,:], y[test_indeces]
model = DEM(layers=32, l1_hidden=0.001, verbose=1)

model.fit(trainX, trainy)
#%%
pred = model.predict(testX)