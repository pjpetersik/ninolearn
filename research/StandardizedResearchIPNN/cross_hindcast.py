import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, norm

from ninolearn.learn.models.ipnn import ipnn
from ninolearn.learn.fit import cross_hindcast
from ninolearn.learn.fit import n_decades, lead_times, decade_color, decade_name

from ninolearn.learn.evaluation import evaluation_correlation, evaluation_decadal_correlation, evaluation_seasonal_correlation
from ninolearn.learn.skillMeasures import inside_fraction_quantiles
from ninolearn.plot.evaluation import plot_seasonal_skill

from ninolearn.IO.read_processed import data_reader
from ninolearn.pathes import processeddir

from cross_training import pipeline

model_name = f'ipnn_new'
cross_hindcast(ipnn, pipeline, model_name)

thresholds = np.arange(-5, 5.25, 0.25)
ds = xr.open_dataset(join(processeddir, f'{model_name}_forecasts.nc')).to_array()

mode_cls = ds.argmax(dim='variable')
mode = mode_cls.copy()
#%%

data = ds.cumsum(dim='variable')

median_cls = (np.abs(data - 0.5).argmin(axis=0))
median = median_cls.copy()

stdm1_cls = (np.abs(data - 0.16).argmin(axis=0))
stdm1 = stdm1_cls.copy()

stdp1_cls = (np.abs(data - 0.84).argmin(axis=0))
stdp1 = stdp1_cls.copy()

stdm2_cls = (np.abs(data - 0.025).argmin(axis=0))
stdm2 = stdm2_cls.copy()

stdp2_cls = (np.abs(data - 0.975).argmin(axis=0))
stdp2 = stdp2_cls.copy()

for i in range(len(thresholds-1)):
    median = median.where(median_cls!=i, other=thresholds[i] + 0.125)
    stdm1 = stdm1.where(stdm1_cls!=i, other=thresholds[i] + 0.125)
    stdp1 = stdp1.where(stdp1_cls!=i, other=thresholds[i] + 0.125)
    stdm2 = stdm2.where(stdm2_cls!=i, other=thresholds[i] + 0.125)
    stdp2 = stdp2.where(stdp2_cls!=i, other=thresholds[i] + 0.125)
    mode = mode.where(mode_cls!=i, other=thresholds[i] + 0.125)


ds_save = xr.Dataset({'median': median,
                 'stdm1': stdm1,
                 'stdp1': stdp1,
                 'stdm2': stdm2,
                 'stdp2': stdp2})

ds_save.to_netcdf(join(processeddir, f'{model_name}_prob_forecasts.nc'))

start = '1963-01'
end = '2017-12'
reader = data_reader(startdate=start, enddate=end)
oni = reader.read_csv('oni')



#%% =============================================================================
# Plot Hindcasts
# =============================================================================
def plot_timeseries(lead, ax):
    ax.axhline(0, c='grey', linestyle='--')
    ax.plot(oni, 'k', lw=2)
    ax.set_xlim(oni.index[0], oni.index[-1])
    ax.fill_between(data.target_season.values,
                    stdm2.loc[{'lead':lead}],
                    stdp2.loc[{'lead':lead}], facecolor='green', alpha=0.2)

    spread = stdp2 - stdm2
    spread_mean = np.round(spread.loc[{'lead':lead}].values.mean(),decimals=1)

    ax.fill_between(data.target_season.values,
                    stdm1.loc[{'lead':lead}],
                    stdp1.loc[{'lead':lead}], facecolor='green', alpha=0.4)
    ax.plot(data.target_season.values, median.loc[{'lead':lead}], 'lime')
    ax.plot(data.target_season.values, mode.loc[{'lead':lead}], 'green')

    inside_frac = inside_fraction_quantiles(oni.values,
                                            stdm2.loc[{'lead':lead}],
                                            stdp2.loc[{'lead':lead}])
    inside_frac = np.round(inside_frac*100, decimals=1)

    ax.set_ylabel('ONI [K]')
    ax.set_ylim(-4,4)
    ax.text(oni.index[11], 2.2, f'{lead}-months,', weight='bold', size=10,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 7})

    ax.text(oni.index[-81], 2.2, f'{inside_frac}%, Spread: {spread_mean}K', weight='bold', size=10,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 7})


plt.close("all")

fig, axs = plt.subplots(6, figsize=(7,7))

for i in range(6):
    plot_timeseries(lead_times[i], axs[i])
    if i!=5:
        axs[i].set_xticks([])

plt.tight_layout()

#%%

def plot_timeseries_spread(lead, ax):
    spread_2std = stdp2 - stdm2
    ax.plot(oni.index, spread_2std.loc[{'lead':lead}])
    ax.set_ylim(0,6)


fig, axs = plt.subplots(6, figsize=(7,7))

lead = [0, 3, 6, 9, 12,15]
for i in range(6):
    plot_timeseries_spread(lead[i], axs[i])
    if i!=5:
        axs[i].set_xticks([])

plt.tight_layout()

#%% =============================================================================
# Invertval propabilities
# =============================================================================

data_dem = xr.open_dataset(join(processeddir, f'dem_review_new_forecasts_with_std_estimated.nc'))

prob = np.zeros((len(thresholds)+1, len(oni), len(lead_times)))
prob[:,:,:] = ds.values
prob_clim = np.zeros_like(prob)

prob_clim[0,:,:] = norm.cdf(thresholds[0], data_dem['mean'], data_dem['std_estimate'])
prob_clim[-1,:,:] = 1 - norm.cdf(thresholds[-1], data_dem['mean'], data_dem['std_estimate'])

oni_cls = np.zeros(len(oni), dtype=int)
oni_cls[:] = np.nan

oni_cls[(thresholds[0]>=oni)] = 0
oni_cls[(thresholds[-1]<oni)] = len(thresholds)-1

oni_cls_prob = np.zeros((len(thresholds)+1, len(oni)))

for k in range(1, len(thresholds)-1):
    prob_clim[k,:,:] = norm.cdf(thresholds[k], data_dem['mean'], data_dem['std_estimate']) \
                        - norm.cdf(thresholds[k-1], data_dem['mean'], data_dem['std_estimate'])

    oni_cls[(thresholds[k-1]<oni)&(thresholds[k]>=oni)] = k

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
# All season correlation skill
# =============================================================================
# scores on the full time series
r, p  = evaluation_correlation(f'{model_name}_prob', variable_name=f'median')

# score in different decades
r_dec, p_dec = evaluation_decadal_correlation(f'{model_name}_prob',variable_name=f'median')

# plot correlation skills
ax = plt.figure(figsize=(6.5,3.5)).gca()

for j in range(n_decades-1):
    plt.plot(lead_times, r_dec[:,j], c=decade_color[j], label=f"Deep Ens.  ({decade_name[j]})")
plt.plot(lead_times, r, label="Deep Ens.  (1963-2017)", c='k', lw=2)

plt.ylim(0,1)
plt.xlim(0.,lead_times[-1])
plt.xlabel('Lead Time [Months]')
plt.ylabel('r')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()


#%% =============================================================================
# Seasonal correlation skills
# =============================================================================
# evaluate the model in different seasons

background = "all"
#background = "el-nino-like"

r_seas, p_seas = evaluation_seasonal_correlation(f'{model_name}_prob', background=background, variable_name=f'median')

plot_seasonal_skill(lead_times, r_seas,  vmin=0, vmax=1)
plt.contour(np.arange(1,13),lead_times, p_seas, levels=[0.01, 0.05, 0.1], linestyles=['solid', 'dashed', 'dotted'], colors='k')
plt.title('Correlation skill')
plt.tight_layout()


#%%
pca_dec = reader.read_statistic('pca', variable='dec_sst', dataset='ERSSTv5', processed='anom')['pca1']
lead = 1
pca_dec_yearly = pca_dec.resample('Y').mean()
yearly_uncertainty =(stdp1[:,lead] - stdm1[:,lead]).resample(target_season='Y', label='right').mean()

print(pearsonr(pca_dec_yearly, yearly_uncertainty))

#%%

plt.subplots(figsize=(12,2))
plt_ds = ds[:,:,5].values

#plt_ds[plt_ds<0.01]=np.nan

plt.contourf(oni.index, thresholds, plt_ds, cmap=plt.cm.Blues, vmin=0.01, vmax=0.3, levels=np.arange(0, 0.5, 0.01))
plt.plot(oni, 'k', lw=2)
plt.tight_layout()



#%%

plt.subplots()
height= ds[:,:, 5].values.mean(axis=1)*4
plt.hist(oni, density=True, bins=thresholds)
plt.bar(thresholds+0.125, height, width=0.25,color='orange', alpha=0.5)
