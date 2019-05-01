"""
here I want to implement the code for the MLP regression and classification
"""

import numpy as np

def include_time_lag(X, max_lag=0):
    Xnew = np.copy(X[max_lag:])
    for i in range (0, max_lag):
        Xnew = np.concatenate((Xnew, X[max_lag-i-1:-i-1]), axis=1)
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