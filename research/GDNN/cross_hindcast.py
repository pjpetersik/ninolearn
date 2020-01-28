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

from cross_training import pipeline
#%%
plt.close("all")

model_name = 'gdnn_ex_pca'
#cross_hindcast(DEM, pipeline, f'{model_name}')

start = '1963-01'
end = '2017-12'
reader = data_reader(startdate=start, enddate=end)
oni = reader.read_csv('oni')

#%% Make a std estimate for comparison purposes
data = xr.open_dataset(join(processeddir,
                      f'{model_name}_forecasts.nc'))

std_estimate = np.zeros((len(oni), len(lead_times)))

for lead in range(len(lead_times)):
    for month in range(0, 12):
        for j in range(n_decades-1):

            time = oni.index

            # test indices
            test_indeces = (time>=f'{decades[j]}-01-01') & (time<=f'{decades[j+1]-1}-12-01')
            train_indeces = np.invert(test_indeces)

            ytrue = oni[train_indeces]
            pred = data['mean'][train_indeces].loc[{'lead':lead_times[lead]}]

            std_seasonal = ytrue.groupby(ytrue.index.month).std()

            r = np.corrcoef(ytrue, pred)[0,1]

            std = std_seasonal.values[month] * ( 1 - r**2)**0.5

            std_estimate[month::12, lead][test_indeces[month::12]] = std

data=data.assign({'std_estimate':(('target_season', 'lead'),std_estimate)})


data.to_netcdf(join(processeddir, f'{model_name}_forecasts_with_std_estimated.nc'))
data.close()
#%% =============================================================================
# Invertval propabilities
# =============================================================================
data = xr.open_dataset((join(processeddir, f'{model_name}_forecasts_with_std_estimated.nc')))

intervals = np.arange(-3, 3.,0.5)

prob = np.zeros((len(intervals) + 1, len(oni), len(lead_times)))
prob_clim = np.zeros_like(prob)
prob[0,:,:] = norm.cdf(intervals[0], data['mean'], data['std'])
prob[-1,:,:] = 1 - norm.cdf(intervals[-1], data['mean'], data['std'])

oni_cls = np.zeros(len(oni), dtype=int)
oni_cls[:] = np.nan

oni_cls[(intervals[0]>=oni)] = 0
oni_cls[(intervals[-1]<oni)] = len(intervals)-1

oni_cls_prob = np.zeros((len(intervals) + 1, len(oni)))

for k in range(1, len(intervals)-1):
    prob[k,:,:] = norm.cdf(intervals[k+1], data['mean'], data['std']) - norm.cdf(intervals[k], data['mean'], data['std'])
    prob_clim[k,:,:] = norm.cdf(intervals[k+1], data['mean'], data['std_estimate']) - norm.cdf(intervals[k], data['mean'], data['std_estimate'])
    oni_cls[(intervals[k]<oni)&(intervals[k+1]>=oni)] = k

for i in range(len(oni)):
    oni_cls_prob[oni_cls[i], i] = 1


rps_mean = np.zeros(len(lead_times))
rps_mean_clim = np.zeros(len(lead_times))

for j in range(len(lead_times)):
    rps_mean[j] =  (((prob[:,:,j] -  oni_cls_prob)**2)).sum(axis=0).mean()
    rps_mean_clim[j] =  (((prob_clim[:,:,j] -  oni_cls_prob)**2)).sum(axis=0).mean()

rpss = 1 - rps_mean/rps_mean_clim
print(rpss)

#%% =============================================================================
# Quantile Score
# =============================================================================
def q_score(q, ytrue, ypred):
    """
    The qunatile score
    """
    e = (ytrue-ypred)

    return np.sum(np.maximum(q*e, (q-1)*e), axis=-1)

start_sel = '1982-01-01'
end_sel = '2017-12-01'
data_sel = data.loc[{'target_season':slice(start_sel, end_sel)}]
ytrue = oni.loc[start_sel : end_sel]

std_levels =[-1.959964, -0.994458, 0.994458, 1.959964]
q_set = np.array([0.025, 0.16, 0.84, 0.975])

qss_estimate = np.zeros((len(q_set), len(lead_times)))


for i in range(len(q_set)):
    for j in range(len(lead_times)):

        q_estimate = data_sel['mean'] + std_levels[i] * data_sel['std_estimate']
        q_gdnn = data_sel['mean'] + std_levels[i] * data_sel['std']

        qs = q_score(q_set[i], ytrue.values, q_gdnn.loc[{'lead':lead_times[j]}])
        qs_estimate = q_score(q_set[i], ytrue.values, q_estimate.loc[{'lead':lead_times[j]}])

        qss_estimate[i,j] = 1 - qs/qs_estimate

plt.figure(figsize=(4,3))
M=plt.pcolormesh(qss_estimate.T, vmin=-0.2, vmax=0.2, cmap=plt.cm.PuOr)
plt.colorbar(M, extend='both')
plt.yticks(np.arange(0.5,8.5), labels=lead_times)
plt.ylabel('Lead')
plt.xticks(np.arange(0.5,4.5), labels=q_set*100)
plt.xlabel('Quantile')
plt.tight_layout()

#%% =============================================================================
# Reliabilty diagramm
# =============================================================================

std_levels =[-1.959964, -0.994458, 0., 0.994458, 1.959964]
q_set = np.array([0.025, 0.16, 0.5, 0.84, 0.975])

def rel_diagram(lead, mean, std):
    frac = np.zeros_like(q_set)
    for i in range(len(q_set)):
        quantile_value_estimate =  mean + std_levels[i] * std
        frac[i] = below_fraction_quantiles(ytrue, quantile_value_estimate) #- q_set[i]
    return frac
plt.figure()
for lead in lead_times:


    plt.plot(q_set, rel_diagram(lead,
                                data_sel['mean'].loc[{'lead':lead}],
                                data_sel['std_estimate'].loc[{'lead':lead}]), ls=':', marker="o")
    plt.plot(q_set, rel_diagram(lead,
                                data_sel['mean'].loc[{'lead':lead}],
                                data_sel['std'].loc[{'lead':lead}]), ls='-', marker="x")

plt.xlim(0,1)
plt.ylim(0,1)
plt.plot([0,1], [0,1], 'k')
plt.xlabel('Predicted Quantile')
plt.xlabel('Observer Frequency')
#%% =============================================================================
# Plot Hindcasts
# =============================================================================
def plot_timeseries(lead, ax):
    ax.axhline(0, c='grey', linestyle='--')
    ax.plot(oni, 'k')
    ax.set_xlim(oni.index[0], oni.index[-1])
    plot_prediction(data.target_season.values, data['mean'].loc[{'lead':lead}],
                    data['std'].loc[{'lead':lead}], ax=ax)

    inside_frac68 = inside_fraction(oni, data['mean'].loc[{'lead':lead}],
                                  data['std'].loc[{'lead':lead}], std_level=1) * 100
    inside_frac68 = np.round(inside_frac68, decimals=1)

    inside_frac95 = inside_fraction(oni, data['mean'].loc[{'lead':lead}],
                                  data['std'].loc[{'lead':lead}], std_level=1.96) * 100
    inside_frac95 = np.round(inside_frac95, decimals=1)


    ax.set_ylabel('ONI [K]')
    ax.set_ylim(-4,4)
    ax.text(oni.index[-63], 2.2, f'{inside_frac68}%', weight='bold', size=10,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 7})

    ax.text(oni.index[-140], 2.2, f'{inside_frac95}%', weight='bold', size=10,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 7})

    ax.text(oni.index[11], 2.2, f'{lead}-month', weight='bold', size=10,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 7})


fig, axs = plt.subplots(8, figsize=(7,10))

for i in range(8):
    plot_timeseries(lead_times[i], axs[i])

    #axs[i].set_xticks([])
    #plt.grid()

plt.tight_layout()

fig, ax = plt.subplots(1, figsize=(7,1.6))
plot_timeseries(lead_times[3], ax)
plt.tight_layout()


#%% =============================================================================
# Spread
# =============================================================================
def plot_timeseries_spread(lead, ax):
    spread_2std_estimate = data['std_estimate'].loc[{'lead':lead}].values *1.96*2
    spread_2std =  data['std'].loc[{'lead':lead}].values *1.96*2

    ax.plot(oni.index,spread_2std)
    ax.plot(oni.index,spread_2std_estimate)
    ax.set_ylim(0,6)
    ax.set_xlim(oni.index[0], oni.index[-1])

fig, axs = plt.subplots(6, figsize=(7,7))

for i in range(6):
    plot_timeseries_spread(lead_times[i], axs[i])
    if i!=5:
        axs[i].set_xticks([])

plt.tight_layout()

#%% =============================================================================
# All season NLL
# =============================================================================
# scores on the full time series

nll  = evaluation_nll(f'{model_name}', filename=f'{model_name}_forecasts_with_std_estimated.nc')
nll_estimate  = evaluation_nll(f'{model_name}', std_name='std_estimate',
                               filename=join(processeddir, f'{model_name}_forecasts_with_std_estimated.nc'))

start='1982-01'
nll82  = evaluation_nll(f'{model_name}', start=start, filename=f'{model_name}_forecasts_with_std_estimated.nc')
nll82_estimate  = evaluation_nll(f'{model_name}', std_name='std_estimate', start=start,
                               filename=join(processeddir, f'{model_name}_forecasts_with_std_estimated.nc'))

# plot correlation skills
ax = plt.figure(figsize=(6.5,3.5)).gca()

plt.plot(lead_times, nll, label="GDNN (1963-2017)", c='k', lw=1)
plt.plot(lead_times, nll_estimate, label=r"GDNN, $\sigma_{clim}$  (1963-2017)", c='k', ls='--', lw=1)

plt.plot(lead_times, nll82, label="GDNN  (1982-2017)", c='r', lw=1)
plt.plot(lead_times, nll82_estimate, label=r"GDNN, $\sigma_{clim}$ (1982-2017)", c='r', ls='--', lw=1)

plt.ylim(-0.7,0.3)
plt.xlim(0.,lead_times[-1])
plt.xlabel('Lead Time [Months]')
plt.ylabel('r')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

#%% =============================================================================
# All season correlation skill
# =============================================================================
# scores on the full time series
r, p  = evaluation_correlation(f'{model_name}')

# score in different decades
r_dec, p_dec = evaluation_decadal_correlation(f'{model_name}')

# plot correlation skills
ax = plt.figure(figsize=(6.5,3.5)).gca()

for j in range(n_decades-1):
    plt.plot(lead_times, r_dec[:,j], c=decade_color[j], label=f"{decade_name[j]}")
plt.plot(lead_times, r, label="1963-2017", c='k', lw=2)

plt.ylim(0,1)
plt.xlim(0.,lead_times[-1])
plt.xlabel('Lead Time [Months]')
plt.ylabel('r')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

# Comapre Ham 2019
r, p  = evaluation_correlation(f'{model_name}', start="1984-01")
ax = plt.figure(figsize=(6.5,3.5)).gca()

plt.plot(np.array(lead_times) + 1.5, r, label="GDNN ens.  (1984-2017)", c='red', lw=2, marker="o")
plt.hlines(0.5,0,50)
plt.ylim(0.25,0.95)
plt.xlim(0.,lead_times[-1]+ 3)
plt.xlabel('Lead Time [Months]')
plt.ylabel('r')
plt.grid()

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

#%% =============================================================================
# Seasonal correlation skills
# =============================================================================
# evaluate the model in different seasons

background = "all"
#background = "la-nina-like"
#background = "el-nino-like"

r_seas, p_seas = evaluation_seasonal_correlation(f'{model_name}',
                                                 background=background,
                                                 variable_name=f'mean')
# mask p-values
p_seasm = np.ma.masked_greater_equal(p_seas, 0.05)

plt.figure(figsize=(6,3))
M=plt.pcolormesh(r_seas, vmin=0.0, vmax=1, cmap=plt.cm.RdYlGn_r, hatch='/')
plt.colorbar(M, extend='min')

plt.pcolor(p_seasm, vmin=0.05, vmax=1, hatch='/',alpha=0.)

plt.yticks(ticks=np.arange(0.5,8.5), labels=lead_times)
plt.xticks(ticks=np.arange(0.5,12.5), labels=seas_ticks)
plt.ylabel('Lead Time [Months]')
plt.xlabel('Target season')
plt.tight_layout()
