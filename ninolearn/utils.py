import numpy as np


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


