import numpy as np

from sklearn.metrics import mean_squared_error

from ninolearn.utils import scale
from scipy.stats import pearsonr


# =============================================================================
# Correlation score
# =============================================================================

def seasonal_correlation(y, pred, time):
    """
    Pearson correlation coefficient for each season. This function uses the\
    scipy.stats.pearsonr function.

    :type y: array_like
    :param y: The true observation.

    :type pred: array_like
    :param pred: The prediction.

    :type y: array_like
    :param y: The corresponding time array for the target season.

    :rtype: array_like
    :return: r, p. Returns the Pearson correlation coefficent (r) and the\
    correspondarray of p-value. Both have length 12 (values for each season).
    """
    r = np.zeros(12)
    p = np.zeros(12)
    for i in range(12):
        month = (time.month == i+1)
        y_sel = scale(y[month])
        pred_sel = scale(pred[month])
        r[i], p[i] = pearsonr(y_sel, pred_sel)
    return r, p


# =============================================================================
#  SRMSE score
# =============================================================================

def rmse(y, predict):
    """
    The root-mean-squarred error (RMSE) for a given observation and prediction.

    :type y: array_like
    :param y: The true observation.

    :type pred: array_like
    :param pred: The prediction.

    :rtype: float
    :return: The RMSE value
    """
    return np.sqrt(mean_squared_error(y, predict))

def seasonal_srmse(y, pred, time):
    """
    Standardized RMSE (RMSE) for each season. Standardized means in this case\
    that the RMSE is divided by the standard deviation of the correpsonding\
    season.

    :type y: array_like
    :param y: The true observation.

    :type pred: array_like
    :param pred: The prediction.

    :type y: array_like
    :param y: The corresponding time array for the target season.

    :rtype: array_like
    :return: Returns the SRMSE for each season. Array has length 12 (value for\
    each season).
    """
    SRMSE = np.zeros(12)

    for i in range(12):
        month = (time.month == i+1)
        y_sel = y[month]
        pred_sel = pred[month]
        SRMSE[i] = np.sqrt(mean_squared_error(y_sel, pred_sel))/np.std(y_sel)
    return SRMSE

def mean_srmse(y, predict, time):
    """
    Mean SRMSE.


    :type y: array_like
    :param y: The true observation.

    :type pred: array_like
    :param pred: The prediction.

    :type y: array_like
    :param y: The corresponding time array for the target season.

    :rtype: float
    :return: The mean SRMSE value.
    """

    seasonal_SRMSE = seasonal_srmse(y, predict, time)
    return np.mean(seasonal_SRMSE)

# =============================================================================
# NEGATIVE LOG-LIKELIHOOD SCORE
# =============================================================================

def nll_gaussian(y, pred_mean, pred_std):
    """
    Negative - log -likelihood for the prediction of a gaussian probability
    """

    first  =  0.5 * np.log(np.square(pred_std))
    second =  np.square(y - pred_mean) / (2  * np.square(pred_std))
    summed = first + second

    nll =  np.mean(summed, axis=-1)
    return nll

def seasonal_nll(y, pred_mean, pred_std, time):
    """
    Negative log-likelihood (NLL) for each season.

    :type y: array_like
    :param y: The true observation.

    :type pred: array_like
    :param pred: The prediction.

    :type y: array_like
    :param y: The corresponding time array for the target season.

    :rtype: array_like
    :return: Returns the NLL for each season. Array has length 12\
    (value for each season).
    """
    score = np.zeros(12)
    for i in range(12):
        month = (time.month == i+1)
        y_sel = y[month]
        pred_mean_sel = pred_mean[month]
        pred_std_sel = pred_std[month]
        score[i] = nll_gaussian(y_sel, pred_mean_sel, pred_std_sel)
    return score

# =============================================================================
# INSIDE MARGINS
# =============================================================================

def inside_fraction(y, pred_mean, pred_std, std_level=1):
    """
    Returns the fraction of how much of the true observation is in the\
    confindence interval.

    :type y: array_like
    :param ytrue: The true observation.

    :type pred_mean: array_like
    :param pred_mean: The mean of the prediction.

    :type pred_std: array_like
    :param pred_std: The standard deviation of the prediction.

    :type std_level: int
    :param std_level: The standard deviation of the confidence interval.

    :rtype: float
    :return: The fraction  of the observation that is in the confidence\
    interval.
    """
    ypred_max = pred_mean + pred_std * std_level
    ypred_min = pred_mean - pred_std * std_level

    in_or_out = np.zeros((len(pred_mean)))
    in_or_out[(y>ypred_min) & (y<ypred_max)] = 1
    in_frac = np.sum(in_or_out)/len(y)

    return in_frac

def below_fraction(y, pred_mean, pred_std, std_level=1):
    """
    Returns the fraction of how much of the true observation is in the\
    confindence interval.

    :type y: array_like
    :param ytrue: The true observation.

    :type pred_mean: array_like
    :param pred_mean: The mean of the prediction.

    :type pred_std: array_like
    :param pred_std: The standard deviation of the prediction.

    :type std_level: int
    :param std_level: The standard deviation of the confidence interval.

    :rtype: float
    :return: The fraction  of the observation that is in the confidence\
    interval.
    """
    ypred_max = pred_mean + pred_std * std_level

    in_or_out = np.zeros((len(pred_mean)))
    in_or_out[y<ypred_max] = 1
    in_frac = np.sum(in_or_out)/len(y)

    return in_frac

def inside_fraction_quantiles(y, low, high):
    """
    Returns the fraction of how much of the true observation is in the\
    confindence interval.

    :type y: array_like
    :param ytrue: The true observation.

    :type pred_mean: array_like
    :param pred_mean: The mean of the prediction.

    :type pred_std: array_like
    :param pred_std: The standard deviation of the prediction.

    :type std_level: int
    :param std_level: The standard deviation of the confidence interval.

    :rtype: float
    :return: The fraction  of the observation that is in the confidence\
    interval.
    """

    in_or_out = np.zeros((len(y)))
    in_or_out[(y>low) & (y<high)] = 1
    in_frac = np.sum(in_or_out)/len(y)

    return in_frac

def below_fraction_quantiles(y, quantlie_value):
    """
    Returns the fraction of how much of the true observation is in the\
    confindence interval.

    :type y: array_like
    :param ytrue: The true observation.

    :type pred_mean: array_like
    :param pred_mean: The mean of the prediction.

    :type pred_std: array_like
    :param pred_std: The standard deviation of the prediction.

    :type std_level: int
    :param std_level: The standard deviation of the confidence interval.

    :rtype: float
    :return: The fraction  of the observation that is in the confidence\
    interval.
    """

    below = np.zeros((len(y)))
    below[(y<quantlie_value)] = 1
    frac = np.sum(below)/len(y)

    return frac
