import matplotlib.pyplot as plt
from ninolearn.IO.read_post import data_reader
from ninolearn.plot.nino_timeseries import nino_background
from ninolearn.utils import scale
from statsmodels.tsa.stattools import ccf
from scipy.stats import spearmanr
from mpl_toolkits.basemap import Basemap

import numpy as np
import pandas as pd

elnino_ep = np.array([1957, 1965, 1972, 1976, 1982,
             1997, 2016])

elnino_cp = np.array([1953, 1958, 1963, 1968, 1969,
             1977, 1979, 1986, 1987, 1991,
             1994, 2002, 2004, 2006, 2009])

lanina_ep = np.array([1964, 1970, 1973, 1988, 1998,
             2007, 2010])

lanina_cp  = np.array([1954, 1955, 1967, 1971, 1974,
              1975, 1984, 1995, 2000, 2001, 2011])

reader = data_reader(startdate='1960-01', enddate='2017-12')

nino34 = reader.read_csv('nino3.4S')

#GODAS data
var = reader.read_netcdf('uwnd', dataset='NCEP', processed='anom')
#var = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom')
#var = reader.read_netcdf('sshg', dataset='GODAS', processed='anom')
#var = reader.read_netcdf('zos', dataset='ORAS4', processed='anom')

spring = np.array([month in [3, 4, 5] for month in nino34.index.month])
summer = np.array([month in [6, 7, 8] for month in nino34.index.month])
winter = np.array([month in [11, 12] for month in nino34.index.month])
winter_p1 = np.array([month in [1] for month in nino34.index.month])

cp = np.array([year in elnino_cp for year in nino34.index.year])
cp_p1 = np.array([year in elnino_cp + 1 for year in nino34.index.year])

ep = np.array([year in elnino_ep for year in nino34.index.year])
ep_p1 = np.array([year in elnino_ep + 1 for year in nino34.index.year])


winter_cp = (winter & cp) | (winter_p1 & cp_p1)
winter_ep = (winter & ep) | (winter_p1 & ep_p1)

winter_nino = winter_cp | winter_ep
summer_cp = (summer & cp)
summer_ep = (summer & ep)
spring_cp = (spring & cp)
spring_ep = (spring & ep)

index_cp = (winter_nino)
index_ep = (winter_ep)

var_mean_cp = var[index_cp,:,:].mean(dim='time', skipna=True)
var_mean_ep = var[index_ep,:,:].mean(dim='time', skipna=True)
#%%
plt.close("all")
lon2, lat2 = np.meshgrid(var_mean_cp.lon, var_mean_cp.lat)

fig = plt.figure()
plt.title("CP")
m = Basemap(projection='robin', lon_0=180, resolution='c')
x, y = m(lon2, lat2)

m.drawparallels(np.arange(-90., 120., 30.))
m.drawmeridians(np.arange(0., 360., 60.))
m.drawmapboundary(fill_color='white')
m.drawcoastlines()

#norm = cm.colors.Normalize(vmax=-1, vmin=1.)
cmap = plt.cm.bwr

vmax = np.max(np.abs(var_mean_ep))
cs = m.contour(x, y, var_mean_cp, cmap=cmap, vmin=-vmax, vmax=vmax)
m.colorbar(cs)


fig = plt.figure()
plt.title("EP")
m = Basemap(projection='robin', lon_0=180, resolution='c')
x, y = m(lon2, lat2)

m.drawparallels(np.arange(-90., 120., 30.))
m.drawmeridians(np.arange(0., 360., 60.))
m.drawmapboundary(fill_color='white')
m.drawcoastlines()

#norm = cm.colors.Normalize(vmax=-1, vmin=1.)
cmap = plt.cm.bwr

cs2 = m.contour(x, y, var_mean_ep, cmap=cmap, vmin=-vmax, vmax=vmax)
m.colorbar(cs2)