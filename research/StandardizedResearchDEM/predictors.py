import numpy as np
import matplotlib.pyplot as plt

from ninolearn.IO.read_processed import data_reader
from ninolearn.plot.nino_timeseries import nino_background

reader = data_reader(startdate='1960-01', enddate='2017-12')

# indeces
oni = reader.read_csv('oni')

iod = reader.read_csv('iod')
wwv = reader.read_csv('wwv_proxy')/10**12

# seasonal cycle
sc = np.cos(np.arange(len(oni))/12*2*np.pi)

# network metrics
network_ssh = reader.read_statistic('network_metrics', variable='zos', dataset='ORAS4', processed="anom")
c2_ssh = network_ssh['fraction_clusters_size_2']
H_ssh = network_ssh['corrected_hamming_distance']

#wind stress
taux = reader.read_netcdf('taux', dataset='NCEP', processed='anom')

taux_WP = taux.loc[dict(lat=slice(2.5,-2.5), lon=slice(120, 160))]
taux_WP_mean = taux_WP.mean(dim='lat').mean(dim='lon')

# decadel variation of leading eof
pca_dec = reader.read_statistic('pca', variable='dec_sst', dataset='ERSSTv5', processed='anom')['pca1']


def plot_timeseries(data, ax, ylabel):
    ax.plot(data, 'k')
    ax.set_xlim(oni.index[0], oni.index[-1])
    ax.set_ylabel(ylabel, rotation=0, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.get_yaxis().set_label_coords(-0.12, 0.4)
    nino_background(oni, ax=ax)


data = [oni, wwv, iod, c2_ssh, H_ssh, taux_WP_mean.to_series(), pca_dec]
y_label= ['ONI [K]', r'WWV [m$^3 \cdot$ 10$^{12}$]', 'DMI [K]', '$c_2$', '$\mathcal{H}^*$', r'$\tau_{x,WP}$ [m$^2/s^2$]','PC1$_{decadal}$']

plt.close("all")
fig, axs = plt.subplots(7,1,  figsize=(12, 8))

for i in range(7):
    plot_timeseries(data[i], axs[i], y_label[i])
    if i!=6:
        axs[i].set_xticks([])

plt.tight_layout()
