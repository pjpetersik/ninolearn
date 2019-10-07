import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from os.path import join

from ninolearn.learn.models.dem import DEM
from ninolearn.learn.fit import cross_hindcast, n_decades, lead_times, decade_color, decade_name

from ninolearn.learn.evaluation import evaluation_nll, evaluation_decadal_nll
from ninolearn.learn.evaluation import evaluation_correlation, evaluation_decadal_correlation, evaluation_seasonal_correlation
from ninolearn.learn.evaluation import evaluation_srmse, evaluation_decadal_srmse, evaluation_seasonal_srmse

from ninolearn.plot.evaluation import plot_seasonal_skill
from ninolearn.plot.prediction import plot_prediction
from ninolearn.IO.read_processed import data_reader
from ninolearn.pathes import processeddir

from cross_training import pipeline
#%%
#cross_hindcast(DEM, pipeline, 'dem')

start = '1963-01'
end = '2017-12'
reader = data_reader(startdate=start, enddate=end)
oni = reader.read_csv('oni')
data = xr.open_dataset(join(processeddir, f'dem_forecasts.nc'))

# =============================================================================
# Plot Hindcasts
# =============================================================================
def plot_timeseries(lead, ax):
    ax.axhline(0, c='grey', linestyle='--')
    ax.plot(oni, 'k')
    ax.set_xlim(oni.index[0], oni.index[-1])
    plot_prediction(data.target_season.values, data['mean'].loc[{'lead':lead}], data['std'].loc[{'lead':lead}], ax=ax)

    ax.set_ylabel('ONI [K]')
    ax.set_ylim(-4,4)
    ax.text(oni.index[11], 2.2, f'{lead}-months', weight='bold', size=10,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 7})

plt.close("all")
fig, ax = plt.subplots(figsize=(8,2))

plot_timeseries(6, ax)

fig, axs = plt.subplots(6, figsize=(7,7))

lead = [0, 3, 6, 9, 12,15]
for i in range(6):
    plot_timeseries(lead[i], axs[i])
    if i!=5:
        axs[i].set_xticks([])

plt.tight_layout()

#%% =============================================================================
# All season NLL skill
# =============================================================================
# scores on the full time series
nll = evaluation_nll('dem')

# score in different decades
nll_dec = evaluation_decadal_nll('dem')

# plot correlation skills
ax = plt.figure(figsize=(6.5,3.5)).gca()

for j in range(n_decades-1):
    plt.plot(lead_times, nll_dec[:,j], c=decade_color[j], label=f"Deep Ens. ({decade_name[j]})")
plt.plot(lead_times, nll, label="Deep Ens.  (1963-2017)", c='k', lw=2)

plt.ylim(-0.6,0.6)
plt.xlim(0.,lead_times[-1])
plt.xlabel('Lead Time [Months]')
plt.ylabel('NLL')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()


#%% =============================================================================
# All season correlation skill
# =============================================================================
# scores on the full time series
r, p  = evaluation_correlation('dem')

# score in different decades
r_dec, p_dec = evaluation_decadal_correlation('dem')

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
# All season SRMSE skill
# =============================================================================
srmse_dec = evaluation_decadal_srmse('dem')
srmse = evaluation_srmse('dem')

# plot SRMSE skills
ax = plt.figure(figsize=(6.5,3.5)).gca()
for j in range(n_decades-1):
    plt.plot(lead_times, srmse_dec[:,j], c=decade_color[j], label=f"Deep Ens.  ({decade_name[j]})")
plt.plot(lead_times, srmse, label="Deep Ens.  (1962-2017)", c='k', lw=2)

plt.ylim(0,1.5)
plt.xlim(0.,lead_times[-1])
plt.xlabel('Lead Time [Months]')
plt.ylabel('SRMSE')
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

r_seas, p_seas = evaluation_seasonal_correlation('dem', background=background)

plot_seasonal_skill(lead_times, r_seas,  vmin=0, vmax=1)
plt.contour(np.arange(1,13),lead_times, p_seas, levels=[0.01, 0.05, 0.1], linestyles=['solid', 'dashed', 'dotted'], colors='k')
plt.title('Correlation skill')
plt.tight_layout()
plt.savefig('/home/paul/Downloads/test.png', dpi=360)

#
srsme_seas = evaluation_seasonal_srmse('dem', background=background)
plot_seasonal_skill(lead_times, srsme_seas,  vmin=0, vmax=1., cmap=plt.cm.inferno_r, extend='max')
plt.title('SRMSE skill')
plt.tight_layout()
