import numpy as np

from ninolearn.utils import scale


def explained_variance(y, pred, time):
    """
    Returns the explained variance (r^2) for each month in a time series
    """
    r = np.zeros(12)
    rsq = np.zeros(12)

    for i in range(12):
        month = (time.month == i+1)
        y_sel = scale(y[month])
        pred_sel = scale(pred[month])
        r[i] = np.corrcoef(y_sel, pred_sel)[0, 1]
        rsq[i] = round(r[i]**2, 3)
    return rsq
