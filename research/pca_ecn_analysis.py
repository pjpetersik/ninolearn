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

nms = networkMetricsSeries('sst', 'ERSSTv5', processed="anom",
                           threshold=0.99, startyear=1948, endyear=2018,
                           window_size=12, lon_min=120, lon_max=280,
                           lat_min=-30, lat_max=30, verbose=1)
nms.computeTimeSeries()


# %% ==========================================================================
# =============================================================================
# # Plot Network Metrics
# =============================================================================
# =============================================================================

reader = data_reader(startdate='1950-01', enddate='2018-12')

nwm_daily = reader.read_network_metrics('air_daily',
                                        dataset='NCEP',
                                        processed='anom')
nino34 = reader.nino34_anom()

plt.close("all")

plt.figure(figsize=(7, 1.5))
nwm_daily['fraction_clusters_size_2'].plot(c='k')
nino_background(nino34)

"""
'global_transitivity'
'avelocal_transmissivity'
'fraction_clusters_size_2'
'fraction_clusters_size_3'
'fraction_clusters_size_5'
'fraction_giant_component'
'average_path_length'
'hamming_distance'
'corrected_hamming_distance'
"""
