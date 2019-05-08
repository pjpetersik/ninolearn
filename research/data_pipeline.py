import numpy as np
from sklearn.preprocessing import StandardScaler

from ninolearn.learn.mlp import include_time_lag
from ninolearn.IO.read_post import data_reader

def pipeline(lead_time,  return_persistance=False):
    """
    data pipeline for the processing of the data before the model is trained
    """
    reader = data_reader(startdate='1981-01')

    # indeces
    nino34 = reader.read_csv('nino3.4S')
    iod = reader.read_csv('iod')
    wwv = reader.read_csv('wwv')

    # seasonal cycle
    sc = np.cos(np.arange(len(nino34))/12*2*np.pi)

    # network metrics
    network_ssh = reader.read_statistic('network_metrics', variable='sshg', dataset='GODAS', processed="anom")
    c2_ssh = network_ssh['fraction_clusters_size_2']

    # pca
    pca_u = reader.read_statistic('pca', variable='uwnd', dataset='NCEP', processed='anom')
    pca2_u = pca_u['pca2']

    # time lag
    time_lag = 12

    # shift such that lead time corresponds to the definition of lead time
    shift = 3

    # process features
    #feature_unscaled = np.stack((nino34, sc, wwv), axis=1)
    feature_unscaled = np.stack((nino34, sc, wwv, iod, pca2_u, c2_ssh), axis=1)

    scalerX = StandardScaler()
    Xorg = scalerX.fit_transform(feature_unscaled)
    Xorg = np.nan_to_num(Xorg)
    X = Xorg[:-lead_time-shift,:]
    X = include_time_lag(X, max_lag=time_lag)

    # process label
    yorg = nino34.values
    y = yorg[lead_time + time_lag + shift:]

    # get the time axis of the label
    timey = nino34.index[lead_time + time_lag + shift:]

    if return_persistance:
        y_persistance = yorg[time_lag: - lead_time - shift]
        return X, y, timey, y_persistance

    else:
        return X, y, timey