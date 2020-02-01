import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from matplotlib.ticker import MaxNLocator

from ninolearn.learn.models.qnn import qnn
from ninolearn.learn.fit import cross_hindcast
from ninolearn.learn.fit import n_decades, lead_times, decade_color, decade_name, decades

from ninolearn.learn.evaluation import evaluation_correlation, evaluation_decadal_correlation, evaluation_seasonal_correlation
from ninolearn.learn.skillMeasures import inside_fraction_quantiles, below_fraction_quantiles
from ninolearn.plot.evaluation import plot_seasonal_skill, seas_ticks

from ninolearn.IO.read_processed import data_reader
from ninolearn.pathes import processeddir

from cross_training import pipeline
#%%
quantiles = [0.025, 0.16, 0.5, 0.84, 0.975]
model_name = f'qnn_ex_pca_tanh'

start = '1963-01'
end = '2017-12'

first = True
for q in quantiles:
    #cross_hindcast(qnn, pipeline, f'{model_name}{round(q*100)}', q=q)
    if first:
        data = xr.open_dataset(join(processeddir, f'{model_name}{round(q*100)}_forecasts.nc'))
        first=False

    else:
        data_append = xr.open_dataset(join(processeddir, f'{model_name}{round(q*100)}_forecasts.nc'))
        data[f'quantile{q}'] = data_append[f'quantile{q}']

data.to_netcdf(join(processeddir,f'{model_name}_forecasts.nc'))


data = data.loc[{'target_season':slice(start, end)}]

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
            pred = data['quantile0.5'][train_indeces].loc[{'lead':lead_times[lead]}]

            std_seasonal = ytrue.groupby(ytrue.index.month).std()

            r = np.corrcoef(ytrue, pred)[0,1]

            std = std_seasonal.values[month] * ( 1 - r**2)**0.5

            std_estimate[month::12, lead][test_indeces[month::12]] = std

data=data.assign({'std_estimate':(('target_season', 'lead'),std_estimate)})


data.to_netcdf(join(processeddir, f'{model_name}_forecasts_with_std_estimated.nc'))
data.close()
#%% =============================================================================
# Plot Hindcasts
# =============================================================================
data = xr.open_dataset(join(processeddir, f'{model_name}_forecasts_with_std_estimated.nc'))

def plot_timeseries(lead, ax):
    ax.axhline(0, c='grey', linestyle='--')
    ax.plot(oni, 'k', lw=2)
    ax.set_xlim(oni.index[0], oni.index[-1])
    ax.fill_between(data.target_season.values,
                    data['quantile0.025'].loc[{'lead':lead}],
                    data['quantile0.975'].loc[{'lead':lead}], facecolor='red', alpha=0.2)

    ax.fill_between(data.target_season.values,
                    data['quantile0.16'].loc[{'lead':lead}],
                    data['quantile0.84'].loc[{'lead':lead}], facecolor='red', alpha=0.4)

    ax.plot(data.target_season.values, data['quantile0.5'].loc[{'lead':lead}], 'red', lw=2)

    inside_frac68 = inside_fraction_quantiles(oni.values,
                                            data['quantile0.16'].loc[{'lead':lead}],
                                            data['quantile0.84'].loc[{'lead':lead}])
    inside_frac68 = np.round(inside_frac68*100, decimals=1)

    inside_frac95 = inside_fraction_quantiles(oni.values,
                                            data['quantile0.025'].loc[{'lead':lead}],
                                            data['quantile0.975'].loc[{'lead':lead}])
    inside_frac95 = np.round(inside_frac95*100, decimals=1)

    ax.set_ylabel('ONI [K]')
    ax.set_ylim(-4,4)

    ax.text(oni.index[-63], 2.2, f'{inside_frac68}%', weight='bold', size=10,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 7})

    ax.text(oni.index[-140], 2.2, f'{inside_frac95}%', weight='bold', size=10,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 7})

    ax.text(oni.index[11], 2.2, f'{lead}-month', weight='bold', size=10,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 7})


plt.close("all")

fig, axs = plt.subplots(8, figsize=(7,10))

for i in range(8):
    plot_timeseries(lead_times[i], axs[i])

plt.tight_layout()

fig, ax = plt.subplots(1, figsize=(7,1.6))
plot_timeseries(lead_times[3], ax)
plt.tight_layout()


#%% =============================================================================
# Quantile Score
# =============================================================================
def q_score(q, ytrue, ypred):
    """
    The qunatile score
    """
    e = (ytrue-ypred)

    return np.mean(np.maximum(q*e, (q-1)*e), axis=-1)

start_sel = '1982-01-01'
end_sel = '2017-12-01'
data_sel = data.loc[{'target_season':slice(start_sel, end_sel)}]
ytrue = oni.loc[start_sel : end_sel]

std_levels =[-1.959964, -0.994458, 0.994458, 1.959964]
q_set = np.array([0.025, 0.16, 0.84, 0.975])

qss_estimate = np.zeros((len(q_set), len(lead_times)))
qss_clim = np.zeros((len(q_set), len(lead_times)))

for i in range(len(q_set)):
    for j in range(len(lead_times)):
        q_estimate = data_sel['quantile0.5'] + std_levels[i] * data_sel['std_estimate']

        q_qnn = data_sel[f'quantile{q_set[i]}']
        qs = q_score(q_set[i], ytrue.values, q_qnn.loc[{'lead':lead_times[j]}])

        qs_estimate = q_score(q_set[i], ytrue.values, q_estimate.loc[{'lead':lead_times[j]}])
        qss_estimate[i,j] = 1 - qs/qs_estimate



plt.figure(figsize=(4,3))
M=plt.pcolormesh(qss_estimate.T, vmin=-0.2, vmax=0.2, cmap=plt.cm.PuOr)
plt.colorbar(M, extend='both')
plt.yticks(np.arange(0.5,8.5), labels=lead_times)
plt.ylabel('Lead Time [Months]')
plt.xticks(np.arange(0.5,4.5), labels=q_set*100)
plt.xlabel('Quantile')
plt.tight_layout()

#%% =============================================================================
# Reliabilty diagramm
# =============================================================================
def rel_diagram(lead):
    frac = np.zeros_like(q_set)
    for i in range(len(q_set)):
        frac[i] = below_fraction_quantiles (oni.values, data[f'quantile{q_set[i]}'].loc[{'lead':lead}])

    return frac

std_levels =[-1.959964, -0.994458, 0, 0.994458, 1.959964]
q_set = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
frac = rel_diagram(3)

plt.figure(figsize=(5,4))
for lead in lead_times:
    plt.plot(q_set, rel_diagram(lead), ls=':', marker="x",label=f"{lead}-months")

plt.xlim(0,1)
plt.ylim(0,1)
plt.plot([0,1], [0,1], 'k')
plt.xlabel('Predicted quantile')
plt.ylabel('Observed frequency')
plt.legend()
#%%

def plot_timeseries_spread(lead, ax):
    spread_2std =  data['quantile0.975']- data['quantile0.025']
    ax.plot(oni.index, spread_2std.loc[{'lead':lead}])

    spread_simple = data['std_estimate'] * 1.96 * 2
    ax.plot(oni.index, spread_simple.loc[{'lead':lead}])

    ax.set_ylim(0,6)

fig, axs = plt.subplots(6, figsize=(7,7))

lead = [0, 3, 6, 9, 12,15]
for i in range(6):
    plot_timeseries_spread(lead[i], axs[i])
    if i!=5:
        axs[i].set_xticks([])

plt.tight_layout()


#%% =============================================================================
# All season correlation skill
# =============================================================================
# scores on the full time series
r, p  = evaluation_correlation(f'{model_name}', variable_name=f'quantile0.5')

# score in different decades
r_dec, p_dec = evaluation_decadal_correlation(f'{model_name}',variable_name=f'quantile0.5')

# plot correlation skills
ax = plt.figure(figsize=(5.,3.)).gca()

for j in range(n_decades-1):
    plt.plot(lead_times, r_dec[:,j], c=decade_color[j], label=f"{decade_name[j]}")
plt.plot(lead_times, r, label="1963-2017", c='k', lw=2)

plt.ylim(0,1)
plt.xlim(0.,lead_times[-1])
plt.xlabel('Lead Time [Months]')
plt.ylabel('ACC')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()


#%% =============================================================================
# Seasonal correlation skills
# =============================================================================
# evaluate the model in different seasons


background = "la-nina-like"
background = "el-nino-like"
#background = "all"

r_seas, p_seas = evaluation_seasonal_correlation(f'{model_name}',
                                                 background=background,
                                                 variable_name=f'quantile0.5')
# mask p-values
p_seasm = np.ma.masked_greater_equal(p_seas, 0.05)

plt.figure(figsize=(5.5,2.8))
M=plt.pcolormesh(r_seas, vmin=0.0, vmax=1, cmap=plt.cm.RdYlGn_r, hatch='/')
plt.colorbar(M, extend="min")

plt.pcolor(p_seasm, vmin=0.05, vmax=1, hatch='/',alpha=0.)

plt.yticks(ticks=np.arange(0.5,8.5), labels=lead_times)
plt.xticks(ticks=np.arange(0.5,12.5), labels=seas_ticks)
plt.ylabel('Lead Time [Months]')
plt.xlabel('Target season')
plt.tight_layout()