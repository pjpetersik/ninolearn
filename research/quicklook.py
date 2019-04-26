import matplotlib.pyplot as plt
from ninolearn.IO.read_post import data_reader
from ninolearn.plot.nino_timeseries import nino_background
from ninolearn.utils import scale
from statsmodels.tsa.stattools import ccf

plt.close("all")

reader = data_reader(startdate='1981-01', enddate='2017-12')
iod = reader.read_csv('iod')
wwv = reader.read_csv('wwv')

nino34 = reader.read_csv('nino3.4M')
nino12 = reader.read_csv('nino1+2M')
nino4 = reader.read_csv('nino4M')

network = reader.read_statistic('network_metrics', variable='sshg',
                           dataset='GODAS', processed="anom")

#network = reader.read_statistic('network_metrics', variable='air',
#                           dataset='NCEP', processed="anom")

pca = reader.read_statistic('pca', variable='air',
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
var = scale(iod)
nino = scale(nino34)

var.plot(c='r')
nino.plot(c='k')

plt.subplots()
plt.xcorr(nino, var, maxlags=80)
plt.vlines(12,-1,1, colors="r")
plt.vlines(6,-1,1, colors="b")
plt.vlines(0,-1,1, colors="k")
plt.ylim(-1,1)
#
#
#%% =============================================================================
# GFDL
# =============================================================================
#reader = data_reader(startdate='2001-01', enddate='2199-12')
#
#nino34gfdl = reader.read_csv('nino3.4M_gfdl')
#
#network = reader.read_statistic('network_metrics', variable='zos',
#                           dataset='GFDL-CM3', processed="anom")
##
##network = reader.read_statistic('network_metrics', variable='tas',
##                           dataset='GFDL-CM3', processed="anom")
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
#pca1 = pca['pca1']
#pca2 = pca['pca2']
#pca3 = pca['pca3']
#
#plt.subplots()
#var = scale(T)
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


#
