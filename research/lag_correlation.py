import matplotlib.pyplot as plt
from ninolearn.IO.read_post import data_reader

from scipy.stats import spearmanr, pearsonr
from ninolearn.private import plotdir

import numpy as np
from os.path import join

def spearman_lag(x,y, max_lags=80):
    scorr = np.zeros(max_lags)
    scorr[0] = spearmanr(x[:], y[:])[0]
    for i in np.arange(1, max_lags):
        scorr[i] = spearmanr(x[i:], y[:-i])[0]

    return scorr

def pearson_lag(x,y, max_lags=28):
    r, p = np.zeros(max_lags+1), np.zeros(max_lags+1)
    r[0], p[0] = pearsonr(x[:], y[:])
    for i in np.arange(1, max_lags+1):
         r[i], p[i] =  pearsonr(x[i:], y[:-i])
    return r, p

def residual(x, y):
    p = np.polyfit(x, y, deg=1)
    ylin = p[0] + p[1] * x
    yres = y - ylin
    return yres

def basin_means(data, lat1=2.5, lat2=-2.5):
    data_basin =  data.loc[dict(lat=slice(lat1, lat2), lon=slice(120, 240))]
    data_basin_mean = data_basin.mean(dim='lat', skipna=True).mean(dim='lon', skipna=True)

    data_WP = data.loc[dict(lat=slice(lat1, lat2), lon=slice(120, 160))]
    data_WP_mean = data_WP.mean(dim='lat', skipna=True).max(dim='lon', skipna=True)

    data_CP = data.loc[dict(lat=slice(lat1, lat2), lon=slice(160, 210))]
    data_CP_mean = data_CP.mean(dim='lat', skipna=True).mean(dim='lon', skipna=True)

    data_EP = data.loc[dict(lat=slice(lat1, lat2), lon=slice(180, 240))]
    data_EP_mean = data_EP.mean(dim='lat', skipna=True).mean(dim='lon', skipna=True)

    return data_basin_mean, data_WP_mean, data_CP_mean, data_EP_mean

plt.close("all")

reader = data_reader(startdate='1960-01', enddate='2017-12', lon_min=30)
nino34 = reader.read_csv('nino3.4S')
wwv = reader.read_csv('wwv_proxy')
taux = reader.read_netcdf('taux', dataset='NCEP', processed='anom')
taux_basin_mean, taux_WP_mean, taux_CP_mean, taux_EP_mean = basin_means(taux, lat1=7.5, lat2=-7.5)
iod = reader.read_csv('iod')

network = reader.read_statistic('network_metrics', variable='zos',
                           dataset='ORAS4', processed="anom")
c2 = network['fraction_clusters_size_2']
H = network['corrected_hamming_distance']


max_lag = 19

r_oni, p_oni = pearson_lag(nino34, nino34, max_lags=max_lag)
r_tau, p_tau = pearson_lag(nino34, taux_WP_mean, max_lags=max_lag)
r_wwv, p_wwv = pearson_lag(nino34, wwv, max_lags=max_lag)
r_iod, p_iod = pearson_lag(nino34, iod, max_lags=max_lag)
r_c2, p_c2 = pearson_lag(nino34, c2, max_lags=max_lag)
r_H, p_H = pearson_lag(nino34, H, max_lags=max_lag)


fig, axs = plt.subplots(1, 2, figsize=(8,3))
axs[0].plot(r_oni)
axs[0].plot(r_tau)
axs[0].plot(r_wwv)
axs[0].plot(r_iod)
axs[0].plot(r_c2)
axs[0].plot(r_H)

axs[0].set_xlim(0, max_lag)
axs[0].set_ylim(-0.8, 1)
axs[0].hlines(0,0, max_lag)
axs[0].set_ylabel('r')
axs[0].set_xlabel('Lag Month')

axs[1].plot(p_oni, label=r'ONI')
axs[1].plot(p_tau, label=r'$tau_{x,WP}$')
axs[1].plot(p_wwv, label='WWV')
axs[1].plot(p_iod, label='DMI (IOD)')
axs[1].plot(p_c2, label=r'c$_2$')
axs[1].plot(p_H, label=r'$\mathcal{H}^*$')



axs[1].set_xlim(0, max_lag)
axs[1].set_ylim(0, 1.)
axs[1].set_ylabel('p-value')
axs[1].set_xlabel('Lag Month')

axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()

plt.savefig(join(plotdir, 'lag_correlation.pdf'))
plt.savefig(join(plotdir, 'lag_correlation.jpg'), dpi=360)

