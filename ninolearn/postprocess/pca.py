from sklearn.decomposition.pca import PCA
from mpl_toolkits.basemap import Basemap
from matplotlib import cm

import matplotlib.pyplot as plt
import numpy as np

from ninolearn.IO.read_post import data_reader


def scaleMax(x):
    return x/np.max(np.abs(x))


reader = data_reader(startdate='1949-01')
data = reader.read_netcdf('air_daily', 'NCEP', 'anom')

time = data['time']
lon = data['lon']
lat = data['lat']
lon2, lat2 = np.meshgrid(lon, lat)

EOFarr = np.array(data[:, :, :])

len_time = len(time)
len_lat = len(lat)
len_lon = len(lon)

EOFarr = EOFarr.reshape((len_time, len_lat * len_lon))

pca = PCA(n_components=6)
pca.fit(EOFarr)

print(pca.explained_variance_ratio_[0:12])


plt.close("all")
fig = plt.figure(figsize=(15, 7))


for i in range(0, 2):
    fig.add_subplot(221+i)
    plt.title("EOF"+str(i+1))
    m = Basemap(projection='robin', lon_0=180, resolution='c')
    x, y = m(lon2, lat2)

    m.drawparallels(np.arange(-90., 120., 30.))
    m.drawmeridians(np.arange(0., 360., 60.))
    m.drawmapboundary(fill_color='white')
    m.drawcoastlines()

    norm = cm.colors.Normalize(vmax=-1, vmin=1.)
    cmap = cm.bwr
    cs = m.pcolormesh(x, y,
                      scaleMax(pca.components_[i, :].reshape(len_lat,
                                                             len_lon)),
                      cmap=cmap, norm=norm)

    cb = m.colorbar(cs)

for i in range(0, 2):
    fig.add_subplot(223+i)
    projection = np.matmul(EOFarr, pca.components_[i, :])

    plt.plot(time, projection)
