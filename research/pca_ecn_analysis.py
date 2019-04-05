import matplotlib.pyplot as plt

from ninolearn.IO.read_post import data_reader
from ninolearn.plot.nino_timeseries import nino_background
from ninolearn.utils import print_header
from ninolearn.postprocess.pca import pca
from ninolearn.postprocess.network import networkMetricsSeries
# =============================================================================
# =============================================================================
# # Compute PCA
# =============================================================================
# =============================================================================
pca_sat = pca(n_components=6)
pca_sat.load_data('air', 'NCEP', processed="anom",
                  startyear=1948, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)
pca_sat.compute_pca()
pca_sat.save()

pca_sat = pca(n_components=6)
pca_sat.load_data('uwnd', 'NCEP', processed="anom",
                  startyear=1948, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)
pca_sat.compute_pca()
pca_sat.save()

pca_sat = pca(n_components=6)
pca_sat.load_data('vwnd', 'NCEP', processed="anom",
                  startyear=1948, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)
pca_sat.compute_pca()
pca_sat.save()


# %% ==========================================================================
# =============================================================================
# # Compute Network Metrics
# =============================================================================
# =============================================================================
print_header("Network Metrics")

nms = networkMetricsSeries('air', 'NCEP', processed="anom",
                           edge_density=0.002, startyear=1948, endyear=2018,
                           window_size=12, lon_min=120, lon_max=280,
                           lat_min=-30, lat_max=30, verbose=1)
nms.computeTimeSeries()


# %% ==========================================================================
# =============================================================================
# # Plot Network Metrics
# =============================================================================
# =============================================================================

reader = data_reader(startdate='1950-01', enddate='2018-12')

nwm_daily = reader.read_statistic('network_metrics', 'air',
                                        dataset='NCEP',
                                        processed='anom')
nino34 = reader.read_csv('nino34')
#wwv = reader.read_csv('wwv')

plt.close("all")

plt.plot(nino34/max(nino34))
#plt.plot(wwv/max(wwv))

plt.figure(figsize=(7, 1.5))
nwm_daily['threshold'].plot(c='k')
nino_background(nino34, nino_treshold=0.5)



