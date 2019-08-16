import matplotlib.pyplot as plt
from ninolearn.IO.read_post import data_reader
from ninolearn.utils import basin_means

from ninolearn.utils import pearson_lag
from ninolearn.private import plotdir

import numpy as np
from os.path import join


plt.close("all")

start = '2002'
end = '2017'

reader = data_reader(startdate=f'{start}-01', enddate=f'{end}-12', lon_min=30)
oni = reader.read_csv('oni')
wwv = reader.read_csv('wwv_proxy')
taux = reader.read_netcdf('taux', dataset='NCEP', processed='anom')
taux_WP_mean, taux_CP_mean, taux_EP_mean = basin_means(taux, lat1=7.5, lat2=-7.5)
iod = reader.read_csv('iod')

network = reader.read_statistic('network_metrics', variable='zos',
                           dataset='ORAS4', processed="anom")
c2 = network['fraction_clusters_size_2']
H = network['corrected_hamming_distance']


max_lag = 19
lead_time_arr = np.arange(-3, max_lag-2)

r_oni, p_oni = pearson_lag(oni, oni, max_lags=max_lag)
r_tau, p_tau = pearson_lag(oni, taux_WP_mean, max_lags=max_lag)
r_wwv, p_wwv = pearson_lag(oni, wwv, max_lags=max_lag)
r_iod, p_iod = pearson_lag(oni, iod, max_lags=max_lag)
r_c2, p_c2 = pearson_lag(oni, c2, max_lags=max_lag)
r_H, p_H = pearson_lag(oni, H, max_lags=max_lag)


fig, axs = plt.subplots(1, 2, figsize=(8,3))
axs[0].plot(lead_time_arr, r_oni)
axs[0].plot(lead_time_arr, r_tau)
axs[0].plot(lead_time_arr,r_wwv)
axs[0].plot(lead_time_arr,r_iod)
axs[0].plot(lead_time_arr,r_c2)
axs[0].plot(lead_time_arr,r_H)

axs[0].set_xlim(-3, max_lag-3)
axs[0].set_ylim(-0.8, 1)
axs[0].hlines(0,-4, max_lag)
axs[0].set_ylabel('r')
axs[0].set_xlabel('Lead Time [Months]')

axs[1].plot(lead_time_arr,p_oni, label=r'ONI')
axs[1].plot(lead_time_arr,p_tau, label=r'$tau_{x,WP}$')
axs[1].plot(lead_time_arr,p_wwv, label='WWV')
axs[1].plot(lead_time_arr,p_iod, label='DMI (IOD)')
axs[1].plot(lead_time_arr,p_c2, label=r'c$_2$')
axs[1].plot(lead_time_arr,p_H, label=r'$\mathcal{H}^*$')



axs[1].set_xlim(-3, max_lag-3)
axs[1].set_ylim(0, 1.)
axs[1].set_ylabel('p-value')
axs[1].set_xlabel('Lead Time [Months]')

axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()

plt.savefig(join(plotdir, f'lag_correlation_{start[2:]}{end[2:]}.pdf'))
plt.savefig(join(plotdir, f'lag_correlation_{start[2:]}{end[2:]}.jpg'), dpi=360)

