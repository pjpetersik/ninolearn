import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from os.path import join


from ninolearn.learn.models.dem import DEM
from ninolearn.learn.fit import cross_hindcast, n_decades, lead_times, decade_color, decade_name

from ninolearn.learn.evaluation import evaluation_correlation, evaluation_decadal_correlation, evaluation_seasonal_correlation
from ninolearn.learn.evaluation import evaluation_srmse, evaluation_decadal_srmse, evaluation_seasonal_srmse
from ninolearn.learn.skillMeasures import inside_fraction

from ninolearn.plot.evaluation import seas_ticks
from ninolearn.plot.prediction import plot_prediction
from ninolearn.IO.read_processed import data_reader
from ninolearn.pathes import processeddir

from cross_training import pipeline_small
#%%
model_name = 'mlp'

intervals = np.arange(-3, 3.1, 0.1)

#cross_hindcast(DEM, pipeline_small, f'{model_name}')

start = '1963-01'
end = '2017-12'
reader = data_reader(startdate=start, enddate=end)
oni = reader.read_csv('oni')
data = xr.open_dataset(join(processeddir, f'{model_name}_forecasts.nc'))

# Make a std estimate for comparison purposes
data=data.assign({'std_estimate':(('target_season', 'lead'),
                             np.zeros((len(oni), len(lead_times))))})

r, p  = evaluation_correlation(f'{model_name}')
std_seasonal = oni.groupby(oni.index.month).std()
std_estimate = np.zeros((len(r), len(std_seasonal)))

for lead in range(len(lead_times)):
    for month in range(len(std_seasonal)):
        std_estimate[lead, month] = std_seasonal.values[month] * ( 1 - r[lead]**2)**0.5
        data['std_estimate'][month::12, lead] = std_estimate[lead, month]

data.to_netcdf(join(processeddir, f'{model_name}_forecasts_with_std_estimated.nc'))

#%% =============================================================================
# Plot Hindcasts
# =============================================================================
def plot_timeseries(lead, ax):
    ax.axhline(0, c='grey', linestyle='--')
    ax.plot(oni, 'k')
    ax.set_xlim(oni.index[0], oni.index[-1])
    plot_prediction(data.target_season.values, data['mean'].loc[{'lead':lead}],
                    data['std_estimate'].loc[{'lead':lead}], ax=ax,
                    facecolor='limegreen', line_color='green')
    ax.set_ylabel('ONI [K]')
    ax.set_ylim(-4,4)
    ax.text(oni.index[11], 2.2, f'{lead}-months,', weight='bold', size=10,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 7})

    inside_frac = inside_fraction(oni, data['mean'].loc[{'lead':lead}],
                                  data['std_estimate'].loc[{'lead':lead}], std_level=1.96) * 100
    inside_frac = np.round(inside_frac, decimals=1)
    ax.text(oni.index[-81], 2.2, f'{inside_frac}%', weight='bold', size=10,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 7})

plt.close("all")

fig, axs = plt.subplots(8, figsize=(7,7))

for i in range(8):
    plot_timeseries(lead_times[i], axs[i])
    if i!=5:
        axs[i].set_xticks([])

plt.tight_layout()

fig, ax = plt.subplots(1, figsize=(7,2))
plot_timeseries(lead_times[3], ax)
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
srmse_dec = evaluation_decadal_srmse(f'{model_name}')
srmse = evaluation_srmse(f'{model_name}')

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


r_seas, p_seas = evaluation_seasonal_correlation(f'{model_name}',
                                                 background=background,
                                                 variable_name=f'mean')
# mask p-values
p_seasm = np.ma.masked_greater_equal(p_seas, 0.05)

plt.figure()
M=plt.pcolormesh(r_seas, vmin=0.0, vmax=1, cmap=plt.cm.RdYlGn_r, hatch='/')
plt.colorbar(M)

plt.pcolor(p_seasm, vmin=0.05, vmax=1, hatch='/',alpha=0.)

plt.yticks(ticks=np.arange(0.5,8.5), labels=lead_times)
plt.xticks(ticks=np.arange(0.5,12.5), labels=seas_ticks)
plt.ylabel('Lead')
plt.xlabel('Target season')