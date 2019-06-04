import matplotlib.pyplot as plt
from ninolearn.IO.read_post import data_reader
from ninolearn.plot.nino_timeseries import nino_background
from ninolearn.utils import scale
from statsmodels.tsa.stattools import ccf
from scipy.stats import spearmanr

import numpy as np

def spearman_lag(x,y, max_lags=80):
    scorr = np.zeros(max_lags)
    scorr[0] = spearmanr(x[:], y[:])[0]
    for i in np.arange(1, max_lags):
        scorr[i] = spearmanr(x[i:], y[:-i])[0]

    return scorr

def pearson_lag(x,y, max_lags=80):
    pcorr = np.zeros(max_lags)
    pcorr[0] = np.corrcoef(x[:], y[:])[0,1]
    for i in np.arange(1, max_lags):
        pcorr[i] = np.corrcoef(x[i:], y[:-i])[0,1]

    return pcorr

plt.close("all")

reader = data_reader(startdate='1981-01', enddate='2017-12')
iod = reader.read_csv('iod')
wwv = reader.read_csv('wwv')

nino34 = reader.read_csv('nino3.4S')
nino12 = reader.read_csv('nino1+2M')
nino4 = reader.read_csv('nino4M')
nino3 = reader.read_csv('nino3M')

network = reader.read_statistic('network_metrics', variable='sshg',
                           dataset='GODAS', processed="anom")

#network = reader.read_statistic('network_metrics', variable='air',
#                           dataset='NCEP', processed="anom")

pca = reader.read_statistic('pca', variable='uwnd',
                          dataset='NCEP', processed="anom")

c2 = network['fraction_clusters_size_2']
c3 = network['fraction_clusters_size_3']
c5 = network['fraction_clusters_size_5']
S = network['fraction_giant_component']
H = network['corrected_hamming_distance']
T = network['global_transitivity']
C = network['avelocal_transmissivity']
L = network['average_path_length']

pca2 = pca['pca2']

plt.subplots()
var = scale(T)
var2 = scale(c2)
var3 = scale(iod)
nino = scale(nino34)
nino3norm = scale(nino3)
nino4norm = scale(nino4)


nino3.plot()
nino4.plot()

var.plot(c='r')
#nino.plot(c='k')

plt.subplots()
plt.vlines(12,-1,1, colors="grey")
plt.vlines(6,-1,1, colors="grey")
plt.vlines(0,-1,1, colors="grey")
plt.xcorr(nino, var, maxlags=80, label="auto-correlation")
plt.xcorr(nino, var2, maxlags=80, color="b", label="c2")
plt.xcorr(nino, var3, maxlags=80, color="r", label="iod")
plt.ylim(-1,1)
plt.xlim(0,48)
plt.legend()
plt.xlabel('lag month')


"""
Archieved


##%% =============================================================================
## GFDL
## =============================================================================
#reader = data_reader(startdate='1701-01', enddate='2199-12')
#
#nino34gfdl = reader.read_csv('nino3.4M_gfdl')
#iodgfdl = reader.read_csv('iod_gfdl')
#network = reader.read_statistic('network_metrics', variable='tos',
#                           dataset='GFDL-CM3', processed="anom")
#
#pca = reader.read_statistic('pca', variable='tas',
#                           dataset='GFDL-CM3', processed="anom")
#
#c2 = network['fraction_clusters_size_2']
#c3 = network['fraction_clusters_size_3']
#c5 = network['fraction_clusters_size_5']
#S = network['fraction_giant_component']
#H = network['corrected_hamming_distance']
#T = network['global_transitivity']
#C = network['avelocal_transmissivity']
#L = network['average_path_length']
#
#pca2 = pca['pca2']
#
#plt.subplots()
#var = scale(C)
#nino = scale(nino34gfdl)
#
#var.plot(c='r')
#nino.plot(c='k')
#
#plt.subplots()
#plt.xcorr(nino, var, maxlags=80)
#plt.vlines(12,-1,1, colors="r")
#plt.vlines(6,-1,1, colors="b")
#plt.vlines(0,-1,1, colors="k")
#plt.ylim(-1,1)
"""