import matplotlib.pyplot as plt

from ninolearn.IO.read_post import data_reader
from ninolearn.plot.nino_timeseries import nino_background
from ninolearn.utils import print_header
from ninolearn.postprocess.pca import pca
from ninolearn.postprocess.network import networkMetricsSeries

# %% ==========================================================================
# =============================================================================
# # Compute Network Metrics
# =============================================================================
# =============================================================================
print_header("Network Metrics")

nms = networkMetricsSeries('sshg', 'GODAS', processed="anom",
                           threshold=0.9, startyear=1980, endyear=2018,
                           window_size=12, lon_min=120, lon_max=280,
                           lat_min=-30, lat_max=30, verbose=1)
nms.computeTimeSeries()


# %% ==========================================================================
# =============================================================================
# # Plot Network Metrics
# =============================================================================
# =============================================================================
plt.close("all")
reader = data_reader(startdate='1981-01', enddate='2018-12')


nwm = reader.read_statistic('network_metrics', 'sshg',
                                        dataset='GODAS',
                                        processed='anom')
nwm2 = reader.read_statistic('network_metrics', 'air_daily',
                                        dataset='NCEP',
                                        processed='anom')


nino34 = reader.read_csv('nino3.4M')

#plt.close("all")
#
#plt.plot(nino34/max(nino34))
##plt.plot(wwv/max(wwv))
#
#plt.figure(figsize=(7, 1.5))
nwm['fraction_clusters_size_2'].plot(c='k')
#nwm2['fraction_clusters_size_2'].plot(c='g')
nino_background(nino34, nino_treshold=0.5)
