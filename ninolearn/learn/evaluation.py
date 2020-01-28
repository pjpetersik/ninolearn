import numpy as np
import pandas as pd

from ninolearn.utils import print_header

from ninolearn.IO.read_processed import data_reader
from ninolearn.learn.skillMeasures import seasonal_correlation
from ninolearn.learn.skillMeasures import mean_srmse, seasonal_srmse
from ninolearn.learn.skillMeasures import nll_gaussian

from ninolearn.learn.fit import lead_times, n_lead, decades, n_decades
from scipy.stats import pearsonr

# =============================================================================
# ALL SEASONS EVALUATION
# =============================================================================

def evaluation_nll(model_name, mean_name = 'mean', std_name='std', filename=None,
                   start='1963-01', end='2017-12'):
    """
    Evaluate the model using the negativ log-likelihood skill for the full time series.
    """
    reader = data_reader(startdate=start, enddate=end)

    # scores for the full timeseries
    nll = np.zeros(n_lead)

    # ONI observation
    obs = reader.read_csv('oni')

    for i in range(n_lead):
        pred_all = reader.read_forecasts(model_name, lead_times[i], filename=filename)
        mean = pred_all[mean_name]
        std = pred_all[std_name]

        # calculate all seasons scores
        nll[i] = nll_gaussian(obs, mean, std)

    return nll


def evaluation_correlation(model_name, variable_name = 'mean', start='1963-01',
                                   end='2017-12'):
    """
    Evaluate the model using the correlation skill for the full time series.

    :type model_name: str
    :param model_name: The name of the model.

    :type variable_name: str
    :param variable_name: The name of the variable which shell be evaluated\
    against the ONI prediction.

    :returns: The correlation skill for the 0, 3, 6, 9, 12 and 15-month lead\
    time and the corresponding p values.
    """
    reader = data_reader(startdate=start, enddate=end)

    # scores for the full timeseries
    r = np.zeros(n_lead)
    p = np.zeros(n_lead)

    # ONI observation
    obs = reader.read_csv('oni')

    for i in range(n_lead):
        pred_all = reader.read_forecasts(model_name, lead_times[i])
        pred = pred_all[variable_name]

        # calculate all seasons scores
        r[i], p[i] = pearsonr(obs, pred)
    return r, p

def evaluation_srmse(model_name, variable_name = 'mean'):
    """
    Evaluate the model using the standardized root-mean-squarred error (SRMSE)
    for the full time series. Standardized means that the the the RMSE of each
    season is divided by the corresponding standard deviation of the ONI in
    that season (standard deviation has a seasonal cycle). Then, these
    seasonal SRMSE averaged to get the SRMSE of the full time series..

    :type model_name: str
    :param model_name: The name of the model.

    :type variable_name: str
    :param variable_name: The name of the variable which shell be evaluated\
    against the ONI prediction.

    :returns: The standardized RMSE for the 0, 3, 6, 9, 12 and 15-month lead\
    time.
    """
    reader = data_reader(startdate='1963-01', enddate='2017-12')

    # scores for the full timeseries
    srmse = np.zeros(n_lead)

    # ONI observation
    obs = reader.read_csv('oni')

    for i in range(n_lead):
        pred_all = reader.read_forecasts(model_name, lead_times[i])
        pred = pred_all[variable_name]

        srmse[i] = mean_srmse(obs, pred, obs.index - pd.tseries.offsets.MonthBegin(1))

    return srmse


# =============================================================================
# DECADAL EVALUATION
# =============================================================================
def evaluation_decadal_nll(model_name, mean_name = 'mean', std_name='std', filename=None):
    """
    Evaluate the model in the decades 1963-1971, 1972-1981, ..., 2012-2017 \
    using the negative log-likelihood.
    """
    reader = data_reader(startdate='1963-01', enddate='2017-12')

    # decadal scores
    decadal_nll = np.zeros((n_lead, n_decades-1))

    # ONI observation
    obs = reader.read_csv('oni')
    obs_time = obs.index

    for i in range(n_lead):

        pred_all = reader.read_forecasts(model_name, lead_times[i], filename)
        pred_mean = pred_all[mean_name]
        pred_std = pred_all[std_name]

        for j in range(n_decades-1):

            indeces = (obs_time>=f'{decades[j]}-01-01') & (obs_time<=f'{decades[j+1]}-12-01')

            decadal_nll[i, j] = nll_gaussian(obs[indeces], pred_mean[indeces], pred_std[indeces])

    return decadal_nll


def evaluation_decadal_correlation(model_name, variable_name = 'mean', start='1963-01',
                                   end='2017-12'):
    """
    Evaluate the model in the decades 1963-1971, 1972-1981, ..., 2012-2017 using the correlation skill-

    :type model_name: str
    :param model_name: The name of the model.

    :type variable_name: str
    :param variable_name: The name of the variable which shell be evaluated\
    against the ONI prediction.

    :returns: The correlation skill for the 0, 3, 6, 9, 12 and 15-month lead\
    time and the corresponding p values for the respective decades. The\
    returned arrays have the shape (lead time, decades).
    """
    reader = data_reader(startdate=start, enddate=end)

    # decadal scores
    decadal_r = np.zeros((n_lead, n_decades-1))
    decadal_p = np.zeros((n_lead, n_decades-1))

    # ONI observation
    obs = reader.read_csv('oni')
    obs_time = obs.index

    for i in range(n_lead):
        pred_all = reader.read_forecasts(model_name, lead_times[i])
        pred = pred_all[variable_name]

        for j in range(n_decades-1):
            indeces = (obs_time>=f'{decades[j]}-01-01') & (obs_time<=f'{decades[j+1]-1}-12-01')
            decadal_r[i, j], decadal_p[i, j] = pearsonr(obs[indeces].values, pred[indeces].values)

    return decadal_r, decadal_p


def evaluation_decadal_srmse(model_name, variable_name = 'mean', decadal=None):
    """
    Evaluate the model in the decades 1963-1971, 1972-1981, ..., 2012-2017 \
    using the standardized RMSE.

    :type model_name: str
    :param model_name: The name of the model.

    :type variable_name: str
    :param variable_name: The name of the variable which shell be evaluated\
    against the ONI prediction.

    :returns: The SRMSE for the 0, 3, 6, 9, 12 and 15-month lead\
    time respective decades. The returned array has the shape (lead time, \
    decades).
    """
    reader = data_reader(startdate='1963-01', enddate='2017-12')

    # decadal scores
    decadal_srmse = np.zeros((n_lead, n_decades-1))

    # ONI observation
    obs = reader.read_csv('oni')
    obs_time = obs.index

    for i in range(n_lead):

        pred_all = reader.read_forecasts(model_name, lead_times[i])
        pred = pred_all[variable_name]

        for j in range(n_decades-1):

            indeces = (obs_time>=f'{decades[j]}-01-01') & (obs_time<=f'{decades[j+1]}-12-01')

            decadal_srmse[i, j] = mean_srmse(obs[indeces], pred[indeces],
                                             obs.index[indeces] - pd.tseries.offsets.MonthBegin(1))

    return decadal_srmse

# =============================================================================
# SEASONAL EVALUATION
# =============================================================================

def evaluation_seasonal_correlation(model_name, variable_name='mean', background='all'):
    """
    Evaluate the model in different seasons using the correlation skill.

    :type model_name: str
    :param model_name: The name of the model.

    :type variable_name: str
    :param variable_name: The name of the variable which shell be evaluated\
    against the ONI prediction.

    :returns: The correlation skill for different seasons and the \
    0, 3, 6, 9, 12 and 15-month lead time and the corresponding p values for\
    the respective seasons and lead times. The returned arrays have the shape \
    (lead time, season). The season corresponding to the the array entry [:,0]\
    is DJF and to [:,1] is JFM (and so on).
    """
    reader = data_reader(startdate='1963-01', enddate='2017-12')

    # seasonal scores
    seasonal_r = np.zeros((n_lead, 12))
    seasonal_p = np.zeros((n_lead, 12))

    # ONI observation
    oni = reader.read_csv('oni')

    if background=="el-nino-like":
        obs = oni[(oni.index.year>=1982)&(oni.index.year<=2001)]
    elif background=="la-nina-like":
        obs = oni[(oni.index.year<1982)|(oni.index.year>2001)]
    elif background=="barnston_2019":
        obs = oni[(oni.index.year>=1982)|(oni.index.year>2015)]
    elif background=="all":
        obs = oni
    obs_time = obs.index

    for i in range(n_lead):

        pred_all = reader.read_forecasts(model_name, lead_times[i]).loc[{'target_season':obs_time}]
        pred = pred_all[variable_name]

        seasonal_r[i, :], seasonal_p[i, :] = seasonal_correlation(obs, pred, obs_time - pd.tseries.offsets.MonthBegin(1))

    return seasonal_r, seasonal_p


def evaluation_seasonal_srmse(model_name, variable_name='mean', background='all'):
    """
    Evaluate the model in different seasons using the standardized RMSE.

    :type model_name: str
    :param model_name: The name of the model.

    :type variable_name: str
    :param variable_name: The name of the variable which shell be evaluated\
    against the ONI prediction.

    :returns: The SRMSE for different seasons and the \
    0, 3, 6, 9, 12 and 15-month lead times. The returned arrays have the shape \
    (lead time, season). The season corresponding to the the array entry [:,0]\
    is DJF and to [:,1] is JFM (and so on).
    """
    reader = data_reader(startdate='1963-01', enddate='2017-12')

    # seasonal scores
    seas_srmse = np.zeros((n_lead, 12))

    # ONI observation
    oni = reader.read_csv('oni')

    if background=="el-nino-like":
        obs = oni[(oni.index.year>=1982)&(oni.index.year<=2001)]
    elif background=="la-nina-like":
        obs = oni[(oni.index.year<1982)|(oni.index.year>2001)]
    elif background=="all":
        obs = oni

    obs_time = obs.index

    for i in range(n_lead):
        pred_all = reader.read_forecasts(model_name, lead_times[i]).loc[{'target_season':obs_time}]
        pred = pred_all[variable_name]

        seas_srmse[i, :] = seasonal_srmse(obs, pred, obs_time - pd.tseries.offsets.MonthBegin(1))

    return seas_srmse
