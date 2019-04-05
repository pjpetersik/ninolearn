import numpy as np
from scipy import interpolate

def window_warping(ts, window_size = [12, 18] , strength=[0.5, 0.8], amount = 12):
    assert type(ts) is np.ndarray

    dim = len(ts.shape)

    if dim == 1:
        ts_warped = window_warping_single(ts, window_size, strength, amount)

    if dim == 2:
        ts_warped = np.zeros_like(ts)
        for i in range(len(ts_warped[1])):
            ts_warped[:,i] = window_warping_single(ts[:,i], window_size, strength, amount)

    return ts_warped

def window_warping_single(ts, window_size , strength, amount):
    """
    window warping for data augmentation of time series.

    :param ts: the timeseries as a 1D np.ndarray

    :param window_size: Half the size of a window size for the warping.

    :param strength: the strength of the stretching/compressing. Float between
    0 and 1.

    :param amount: The amount how often a random window of the time series
    should be warped
    """
    assert type(ts) is np.ndarray

    len_ts = len(ts)
    x = np.arange(len_ts)
    f = interpolate.interp1d(x, ts)
    new_ts = np.zeros_like(ts)
    new_ts[:] = ts

    if type(window_size) == list:
        ws = np.random.randint(window_size[0], window_size[1])
    else:
        ws = window_size

    if type(strength) == list:
        s = np.random.uniform(strength[0], strength[1])
    else:
        s = strength


    for _ in range(amount):
        middle = np.random.randint(ws,len_ts-ws)
        begin = middle - ws
        end = middle + ws

        x_middle_shifted = x[middle] + ws * s * np.random.choice([-1,1])

        x1= np.linspace(x[begin], x_middle_shifted, ws, endpoint=False)
        x2 = np.linspace(x_middle_shifted, x[end], ws)

        new_ts[begin:middle] = f(x1)
        new_ts[middle:end] = f(x2)

    return new_ts

