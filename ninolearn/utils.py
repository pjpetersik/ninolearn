"""
In this module, methods are collected which have not found a proper module yet
to which they belong. Help them find there home!
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr

def print_header(string):
    print()
    print("##################################################################")
    print(string)
    print("##################################################################")
    print()


def small_print_header(string):
    print(string)
    print("--------------------------------------")


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def lowest_indices(ary, n):
    """Returns the n lowest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(flat[indices])]
    return np.unravel_index(indices, ary.shape)


def generateFileName(variable, dataset, processed='', suffix=None):
    """
    generates a file name
    """
    filenamelist = [variable, dataset, processed]

    # remove ''  entries from list
    filenamelist = list(filter(lambda a: a != '', filenamelist))

    filename = '_'.join(filenamelist)

    if suffix is not None:
        filename = '.'.join([filename, suffix])

    return filename


def scale(x):
    """
    scale a time series
    """
    return (x-x.mean())/x.std()


def scaleMax(x):
    """
    sacle timeseries by absolute maximum
    """
    return x/np.max(np.abs(x))


"""
here I want to implement the code for the MLP regression and classification
"""


def include_time_lag(X, n_lags=0, step=1):
    Xnew = np.copy(X[n_lags*step:])
    for i in range (1, n_lags):
        Xnew = np.concatenate((Xnew, X[(n_lags-i)*step:-i*step]), axis=1)
    return Xnew


def nino_to_category(nino, categories=None, threshold=None):
    """
    This method translates a NINO index value into a category. NOTE: Either the
    categories OR threshold method can be used!

    :param nino: the timeseries of the NINO index.

    :param categories: The number of categories.

    :param threshod: The threshold for the.
    """
    if categories != None and threshold != None:
        raise Exception("Either categories OR threshold method can be used!")

    if threshold == None:
        sorted_arr = np.sort(nino)
        n = len(sorted_arr)
        n_cat = n//categories
        bounds = np.zeros(categories+1)

        for i in range(1,categories):
            bounds[i] = sorted_arr[i*n_cat]
        bounds[0] = sorted_arr[0] -1
        bounds[-1] = sorted_arr[-1]+1

        nino_cat = np.zeros_like(nino, dtype=int) + categories

        for j in range(categories):
            nino_cat[(nino>bounds[j]) & (nino<=bounds[j+1])] = j

        assert (nino_cat != categories).all()
    else:
        nino_cat = np.zeros_like(nino, dtype=int) + 1
        nino_cat[nino>threshold] = 2
        nino_cat[nino<-threshold] = 0
    return nino_cat


def basin_means(data, lat1=2.5, lat2=-2.5):
    """
    Computes the mean in different basins of the equatorial Pacific

    :param data: The data for which the Basins means shall be computed with
    dimension (time, lat, lon).

    :param lat1, lat2: The latidual bounds

    :returns: The mean in the west Pacific (120E- 160E), the central Pacifc (160E-180E),
    east Pacifc  (180E- 240W).
    """
    data_WP = data.loc[dict(lat=slice(lat1, lat2), lon=slice(120, 160))]
    data_WP_mean = data_WP.mean(dim='lat', skipna=True).mean(dim='lon', skipna=True)

    data_CP = data.loc[dict(lat=slice(lat1, lat2), lon=slice(160, 180))]
    data_CP_mean = data_CP.mean(dim='lat', skipna=True).mean(dim='lon', skipna=True)

    data_EP = data.loc[dict(lat=slice(lat1, lat2), lon=slice(180, 240))]
    data_EP_mean = data_EP.mean(dim='lat', skipna=True).mean(dim='lon', skipna=True)

    return data_WP_mean, data_CP_mean, data_EP_mean

def spearman_lag(x, y, max_lags=80):
    """
    Computes the Spearman lag correlation coefficents using  of x and y until a maximum number of lag time
    steps.

    :param x: The variable that leads.

    :param y: The variable that lags.

    :param max_lags: The maximum number of time steps the for which the
    lag correlation is computed.

    :returns: A timeseries with the lag correlations.
    """
    r = np.zeros(max_lags)
    r[0] = spearmanr(x[:], y[:])[0]
    for i in np.arange(1, max_lags):
        r[i] = spearmanr(x[i:], y[:-i])[0]
    return r

def pearson_lag(x, y, max_lags=28):
    """
    Computes the Pearson lag correlation coefficents using  of x and y until a maximum number of lag time
    steps.

    :param x: The variable that leads.

    :param y: The variable that lags.

    :param max_lags: The maximum number of time steps the for which the
    lag correlation is computed.

    :returns: A timeseries with the lag correlations and the corresponding p-value.
    """
    r, p = np.zeros(max_lags+1), np.zeros(max_lags+1)
    r[0], p[0] = pearsonr(x[:], y[:])
    for i in np.arange(1, max_lags+1):
         r[i], p[i] =  pearsonr(x[i:], y[:-i])
    return r, p


