import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from os.path import join
from scipy.stats import norm, pearsonr

from ninolearn.learn.models.dem import DEM
from ninolearn.learn.fit import cross_hindcast, n_decades, decades, lead_times, decade_color, decade_name

from ninolearn.learn.evaluation import evaluation_nll, evaluation_decadal_nll
from ninolearn.learn.evaluation import evaluation_correlation, evaluation_decadal_correlation, evaluation_seasonal_correlation
from ninolearn.learn.evaluation import evaluation_srmse, evaluation_decadal_srmse
from ninolearn.learn.skillMeasures import inside_fraction, below_fraction_quantiles

from ninolearn.plot.evaluation import seas_ticks
from ninolearn.plot.prediction import plot_prediction
from ninolearn.IO.read_processed import data_reader
from ninolearn.pathes import processeddir


plt.close("all")

start = '1963-01'
end = '2017-12'
reader = data_reader(startdate=start, enddate=end)
oni = reader.read_csv('oni')

gdnn_model = 'gdnn_ex_pca'
qrnn_model = 'qnn_ex_pca_tanh'

file_gdnn = join(processeddir, f'{gdnn_model}_forecasts_with_std_estimated.nc')
file_qrnn = join(processeddir, f'{qrnn_model}_forecasts_with_std_estimated.nc')

data_gdnn = xr.open_dataset(file_gdnn)
data_qrnn = xr.open_dataset(file_qrnn)


#%% =============================================================================
# All season correlation skill Figure 2
# =============================================================================

fig, ax = plt.subplots(1,2, figsize=(8.5,3.5))

r_gdnn, _  = evaluation_correlation(f'{gdnn_model}', variable_name=f'mean' )
r_dec_gdnn, _ = evaluation_decadal_correlation(f'{gdnn_model}',variable_name=f'mean')


r_qrnn, _  = evaluation_correlation(f'{qrnn_model}', variable_name=f'quantile0.5' )
r_dec_qrnn, _ = evaluation_decadal_correlation(f'{qrnn_model}',variable_name=f'quantile0.5')


for j in range(n_decades-1):
    ax[0].plot(lead_times, r_dec_gdnn[:,j], c=decade_color[j], label=f"{decade_name[j]}")
    ax[1].plot(lead_times, r_dec_qrnn[:,j], c=decade_color[j], label=f"{decade_name[j]}")


ax[0].plot(lead_times, r_gdnn, label="1963-2017", c='k', lw=2)
ax[1].plot(lead_times, r_qrnn, label="1963-2017", c='k', lw=2)

cap = ['a', 'b']
for i in range(2):
    ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[i].grid()
    ax[i].set_ylim(0,1)
    ax[i].set_xlim(0,lead_times[-1])

    ax[i].set_ylabel('ACC')
    ax[i].set_xlabel('Lead Time [Months]')
    ax[i].text(18, 0.85, f'{cap[i]}', weight='bold', size=14,
        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 9})


plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

#%% =============================================================================
# Seasonal ACC Figure 3
# =============================================================================

background = "la-nina-like"
background = "el-nino-like"

fig, ax = plt.subplots(2,2, figsize=(10.5,6.5))

# GDNN all
background = "all"
r_seas_gdnn, p_seas_gdnn = evaluation_seasonal_correlation(f'{gdnn_model}',
                                                 background=background,
                                                 variable_name=f'mean')
p_seasm = np.ma.masked_greater_equal(p_seas_gdnn, 0.05)
M=ax[0,0].pcolormesh(r_seas_gdnn, vmin=0.0, vmax=1, cmap=plt.cm.RdYlGn_r, hatch='/')
ax[0,0].pcolor(p_seasm, vmin=0.05, vmax=1, hatch='/',alpha=0.)

# QRNN all
background = "all"
r_seas_gdnn, p_seas_gdnn = evaluation_seasonal_correlation(f'{qrnn_model}',
                                                 background=background,
                                                 variable_name=f'quantile0.5')
p_seasm = np.ma.masked_greater_equal(p_seas_gdnn, 0.05)
M=ax[0,1].pcolormesh(r_seas_gdnn, vmin=0.0, vmax=1, cmap=plt.cm.RdYlGn_r, hatch='/')
ax[0,1].pcolor(p_seasm, vmin=0.05, vmax=1, hatch='/',alpha=0.)


# GDNN all
background = "el-nino-like"
r_seas_gdnn, p_seas_gdnn = evaluation_seasonal_correlation(f'{gdnn_model}',
                                                 background=background,
                                                 variable_name=f'mean')
p_seasm = np.ma.masked_greater_equal(p_seas_gdnn, 0.05)
M=ax[1,0].pcolormesh(r_seas_gdnn, vmin=0.0, vmax=1, cmap=plt.cm.RdYlGn_r, hatch='/')
ax[1,0].pcolor(p_seasm, vmin=0.05, vmax=1, hatch='/',alpha=0.)

# GDNN all
background = "la-nina-like"
r_seas_gdnn, p_seas_gdnn = evaluation_seasonal_correlation(f'{gdnn_model}',
                                                 background=background,
                                                 variable_name=f'mean')
p_seasm = np.ma.masked_greater_equal(p_seas_gdnn, 0.05)
M=ax[1,1].pcolormesh(r_seas_gdnn, vmin=0.0, vmax=1, cmap=plt.cm.RdYlGn_r, hatch='/')
ax[1,1].pcolor(p_seasm, vmin=0.05, vmax=1, hatch='/',alpha=0.)



cap = ['a', 'b', 'c', 'd']
for i in range(2):
    for j in range(2):
        ax[i,j].set_yticks(np.arange(0.5,8.5))
        ax[i,j].set_yticklabels(lead_times)
        ax[i,j].set_ylabel('Lead Time [Months]')
        ax[i,j].set_xticks(ticks=np.arange(0.5,12.5))
        ax[i,j].set_xticklabels(seas_ticks)
        ax[i,j].set_xlabel('Target Season')
        ax[i,j].text(0.8, 6.8, f'{cap[i*2+j]}', weight='bold', size=14,
        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 9})

        fig.colorbar(M, ax=ax[i,j], extend='min')

plt.tight_layout()


#%% =============================================================================
# Quantile Score Figure 4
# =============================================================================

std_levels =[-1.959964, -0.994458, 0.994458, 1.959964]
q_set = np.array([0.025, 0.16, 0.84, 0.975])

def q_score(q, ytrue, ypred):
    """
    The qunatile score
    """
    e = (ytrue-ypred)

    return np.sum(np.maximum(q*e, (q-1)*e), axis=-1)

def qss_gdnn(start_sel, end_sel):
    ytrue = oni.loc[start_sel : end_sel]
    qss_estimate = np.zeros((len(q_set), len(lead_times)))
    data_sel = data_gdnn.loc[{'target_season':slice(start_sel, end_sel)}]
    for i in range(len(q_set)):
        for j in range(len(lead_times)):

            q_estimate = data_sel['mean'] + std_levels[i] * data_sel['std_estimate']
            q_gdnn = data_sel['mean'] + std_levels[i] * data_sel['std']

            qs = q_score(q_set[i], ytrue.values, q_gdnn.loc[{'lead':lead_times[j]}])
            qs_estimate = q_score(q_set[i], ytrue.values, q_estimate.loc[{'lead':lead_times[j]}])

            qss_estimate[i,j] = 1 - qs/qs_estimate

    return qss_estimate


def qss_qrnn(start_sel, end_sel):
    ytrue = oni.loc[start_sel : end_sel]
    qss_estimate = np.zeros((len(q_set), len(lead_times)))
    data_sel = data_qrnn.loc[{'target_season':slice(start_sel, end_sel)}]
    for i in range(len(q_set)):
        for j in range(len(lead_times)):
            q_estimate = data_sel['quantile0.5'] + std_levels[i] * data_sel['std_estimate']

            q_qnn = data_sel[f'quantile{q_set[i]}']
            qs = q_score(q_set[i], ytrue.values, q_qnn.loc[{'lead':lead_times[j]}])

            qs_estimate = q_score(q_set[i], ytrue.values, q_estimate.loc[{'lead':lead_times[j]}])
            qss_estimate[i,j] = 1 - qs/qs_estimate

    return qss_estimate


qss_gdnn_arr = qss_gdnn('1963-01-01', '2017-12-01')
qss_gdnn82_arr = qss_gdnn('1982-01-01', '2017-12-01')

qss_qrnn_arr = qss_qrnn('1963-01-01', '2017-12-01')
qss_qrnn82_arr = qss_qrnn('1982-01-01', '2017-12-01')


fig, ax = plt.subplots(2,2, figsize=(8.5,6.5))


M=ax[0,0].pcolormesh(qss_gdnn_arr.T, vmin=-0.2, vmax=0.2, cmap=plt.cm.PuOr)
M=ax[1,0].pcolormesh(qss_gdnn82_arr.T, vmin=-0.2, vmax=0.2, cmap=plt.cm.PuOr)
M=ax[0,1].pcolormesh(qss_qrnn_arr.T, vmin=-0.2, vmax=0.2, cmap=plt.cm.PuOr)
M=ax[1,1].pcolormesh(qss_qrnn82_arr.T, vmin=-0.2, vmax=0.2, cmap=plt.cm.PuOr)

cap = ['a', 'b', 'c', 'd']
for i in range(2):
    for j in range(2):
        ax[i,j].set_yticks(np.arange(0.5,8.5))
        ax[i,j].set_yticklabels(lead_times)
        ax[i,j].set_ylabel('Lead Time [Months]')
        ax[i,j].set_xticks(np.arange(0.5,4.5))
        ax[i,j].set_xticklabels(q_set*100)
        ax[i,j].set_xlabel('Quantile')
        ax[i,j].text(0.3, 6.8, f'{cap[i*2+j]}', weight='bold', size=14,
        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 9})

        fig.colorbar(M, ax=ax[i,j], extend='both')

plt.tight_layout()

