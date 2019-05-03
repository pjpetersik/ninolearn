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


