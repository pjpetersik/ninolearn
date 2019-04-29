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


#%% =============================================================================
# GFDL Data zos
# =============================================================================
nms = networkMetricsSeries('zos', 'GFDL-CM3', processed="anom",
                           threshold=0.9, startyear=1700, endyear=2199,
                           window_size=12, lon_min=120, lon_max=280,
                           lat_min=-30, lat_max=30, verbose=1)
nms.computeTimeSeries()

#%% =============================================================================
# GFDL Data tos
# =============================================================================
nms = networkMetricsSeries('tos', 'GFDL-CM3', processed="anom",
                           threshold=0.9, startyear=1700, endyear=2199,
                           window_size=12, lon_min=120, lon_max=280,
                           lat_min=-30, lat_max=30, verbose=1)
nms.computeTimeSeries()

#%% =============================================================================
# GFDL Data tas
# =============================================================================
nms = networkMetricsSeries('tas', 'GFDL-CM3', processed="anom",
                           threshold=0.9, startyear=1700, endyear=2199,
                           window_size=12, lon_min=120, lon_max=280,
                           lat_min=-30, lat_max=30, verbose=1)
nms.computeTimeSeries()

# %% ==========================================================================
# =============================================================================
# # Plot Network Metrics
# =============================================================================
# =============================================================================
plt.close("all")
var = 'fraction_clusters_size_3'
#var = 'corrected_hamming_distance'
var = 'avelocal_transmissivity'
var = 'average_path_length'
# =============================================================================
# GFDL
# =============================================================================
readergfdl = data_reader(startdate='1701-01', enddate='2199-12')

nwm_gfdl = readergfdl.read_statistic('network_metrics', 'zos',
                                        dataset='GFDL-CM3',
                                        processed='anom')

nino34gfdl = readergfdl.read_csv('nino3.4M_gfdl')


plt.figure(figsize=(7, 1.5))
nwm_gfdl[var].plot(c='k')
nino_background(nino34gfdl, nino_treshold=0.5)

plt.subplots()
plt.xcorr(nino34gfdl, nwm_gfdl[var], maxlags=48)

# =============================================================================
# Observations
# =============================================================================
readerobs = data_reader(startdate='1981-01', enddate='2018-12')
nwm_obs = readerobs.read_statistic('network_metrics', 'sshg',
                                        dataset='GODAS',
                                        processed='anom')

nino34 = readerobs.read_csv('nino3.4M')

plt.figure(figsize=(7, 1.5))
nwm_obs[var].plot(c='k')
nino_background(nino34, nino_treshold=0.5)

plt.subplots()
plt.xcorr(nino34, nwm_obs[var], maxlags=48)
