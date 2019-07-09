import numpy as np
from sklearn.preprocessing import StandardScaler

from ninolearn.learn.mlp import include_time_lag
from ninolearn.IO.read_post import data_reader

def pipeline(lead_time,  return_persistance=False):
    """
    data pipeline for the processing of the data before the model is trained
    """
    reader = data_reader(startdate='1960-01', enddate='2017-12')

    # indeces
    nino34 = reader.read_csv('nino3.4S')

    iod = reader.read_csv('iod')
    wwv = reader.read_csv('wwv_proxy')

    # seasonal cycle
    sc = np.cos(np.arange(len(nino34))/12*2*np.pi)

    # network metrics
    # network_ssh = reader.read_statistic('network_metrics', variable='sshg', dataset='GODAS', processed="anom")
    network_ssh = reader.read_statistic('network_metrics', variable='zos', dataset='ORAS4', processed="anom")
    c2_ssh = network_ssh['fraction_clusters_size_2']
    H_ssh = network_ssh['corrected_hamming_distance']

    #wind stress
    taux = reader.read_netcdf('taux', dataset='NCEP', processed='anom')

    taux_WP = taux.loc[dict(lat=slice(2.5,-2.5), lon=slice(120, 160))]
    taux_WP_mean = taux_WP.mean(dim='lat').mean(dim='lon')

    taux_CP = taux.loc[dict(lat=slice(2.5,-2.5), lon=slice(160, 180))]
    taux_CP_mean = taux_CP.mean(dim='lat').mean(dim='lon')

    taux_EP = taux.loc[dict(lat=slice(2.5,-2.5), lon=slice(180, 240))]
    taux_EP_mean = taux_EP.mean(dim='lat').mean(dim='lon')

    # time lag
    time_lag = 12

    # shift such that lead time corresponds to the definition of lead time
    shift = 3

    # process features
    feature_unscaled = np.stack((nino34, sc, wwv, iod,
                                 taux_WP_mean, #taux_CP_mean, taux_EP_mean,
                                 c2_ssh, H_ssh), axis=1)

    # scale each feature
    scalerX = StandardScaler()
    Xorg = scalerX.fit_transform(feature_unscaled)

    # set nans to 0.
    Xorg = np.nan_to_num(Xorg)

    # arange the feature array
    X = Xorg[:-lead_time-shift,:]
    X = include_time_lag(X, max_lag=time_lag)

    # arange label
    yorg = nino34.values
    y = yorg[lead_time + time_lag + shift:]

    # get the time axis of the label
    timey = nino34.index[lead_time + time_lag + shift:]

    if return_persistance:
        y_persistance = yorg[time_lag: - lead_time - shift]
        return X, y, timey, y_persistance

    else:
        return X, y, timey


class pipeline_ed(object):
    def get_data(self, lead_time, return_persistance=False):
        #read data
        reader = data_reader(startdate='1971-01', enddate='2018-12')
        sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom')
        nino = reader.read_csv('nino3.4S')

        # the doubling of label
        feature = sst
        label = sst

        # preprocess data
        feature_unscaled = feature.values.reshape(feature.shape[0],-1)
        label_unscaled = label.values.reshape(label.shape[0],-1)

        self.scaler_f = StandardScaler()
        Xorg = self.scaler_f.fit_transform(feature_unscaled)

        self.scaler_l = StandardScaler()
        yorg = self.scaler_l.fit_transform(label_unscaled)

        Xall = np.nan_to_num(Xorg)
        yall = np.nan_to_num(yorg)

        lead = 6

        if lead == 0:
            y = yall
            X = Xall
        else:
            y = yall[lead:]
            X = Xall[:-lead]


        timey = nino.index[lead:]

        return X, y, timey