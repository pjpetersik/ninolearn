import matplotlib.pyplot as plt
from ninolearn.IO.read_post import data_reader
from ninolearn.private import plotdir

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from os.path import join

elnino_ep = np.array([1957, 1965, 1972, 1976, 1982, 1997#  #2015
                      ])

elnino_cp = np.array([1953, 1958, 1963, 1968, 1969,
             1977, 1979, 1986, 1987, 1991,
             1994, 2002, 2004, 2006, 2009#, 2015
             ])

lanina_ep = np.array([1964, 1970, 1973, 1988, 1998,
             2007, 2010])

lanina_cp  = np.array([1954, 1955, 1967, 1971, 1974,
              1975, 1984, 1995, 2000, 2001, 2011])

year = 2008
reader = data_reader(startdate=f'1980-01', enddate=f'2010-12', lon_min=30, lon_max=300)

# =============================================================================
# Read data
# =============================================================================
taux = reader.read_netcdf('taux', dataset='NCEP', processed='')
taux = taux.sortby('lat', ascending=False)
tauy = reader.read_netcdf('tauy', dataset='NCEP', processed='')
tauy = tauy.sortby('lat', ascending=False)

sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='')
sst = sst.sortby('lat', ascending=False)
#ssh = reader.read_netcdf('sshg', dataset='GODAS', processed='anom')
#ssh = ssh.sortby('lat', ascending=False)
#olr =  - reader.read_netcdf('olr', dataset='NCAR', processed='anom')
#olr = olr.sortby('lat', ascending=False)


taux_mean = taux[:,:,:].mean(dim='time', skipna=True)
tauy_mean = tauy[:,:,:].mean(dim='time', skipna=True)
sst_mean = sst[:,:,:].mean(dim='time', skipna=True)

#ssh_mean = ssh[index,:,:].mean(dim='time', skipna=True)
#olr_mean = olr[index,:,:].mean(dim='time', skipna=True)


#%% =============================================================================
# #Plots
# =============================================================================
#levels_olr = levels = np.arange(-80.,90., 10)
levels_tau = np.round(np.arange(-155.,155., 5), decimals=1)
#levels_ssh = np.round(np.arange(-1, 1.05, .05), decimals=2)
levels_sst = np.arange(20, 30, 0.5)

plt.close("all")
lon2, lat2 = np.meshgrid(taux_mean.lon, taux_mean.lat)

fig, axs = plt.subplots(figsize=(8,3))


m = Basemap(projection='merc',llcrnrlat=-30,urcrnrlat=30,\
            llcrnrlon=100,urcrnrlon=300,lat_ts=5,resolution='c',ax=axs)

x, y = m(lon2, lat2)

m.drawparallels(np.arange(-90., 120., 15.), labels=[1,0,0,0], color='grey')
m.drawmeridians(np.arange(0., 360., 30.), labels=[0,0,0,1], color='grey')
m.drawmapboundary(fill_color='white')
m.drawcoastlines()

cs_sst = m.contourf(x, y, sst_mean, cmap=plt.cm.seismic,levels=levels_sst, extend='both')

divider = make_axes_locatable(axs)
cax2 = divider.append_axes("right", size="8%", pad=0.1)
cax2.axis('off')
m.colorbar(cs_sst, label=r'SST [$^{\circ}$C]')



yy = np.arange(0, y.shape[0], 3)
xx = np.arange(0, x.shape[1], 3)

points = np.meshgrid(yy, xx)

m.quiver(x[points], y[points], taux_mean.values[points], tauy_mean.values[points])

plt.tight_layout()

plt.savefig(join(plotdir, 'composite_mean.jpg'), dpi=360)