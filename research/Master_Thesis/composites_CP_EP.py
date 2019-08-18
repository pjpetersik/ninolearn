import matplotlib.pyplot as plt
from ninolearn.IO.read_processed import data_reader
from ninolearn.private import plotdir

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from os.path import join

save = False

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

reader = data_reader(startdate='1974-06', enddate='2017-12', lon_min=30, lon_max=300)

oni = reader.read_csv('oni')

spring = np.array([month in [1,2, 3, ] for month in oni.index.month])
summer = np.array([month in [6, 7, 8] for month in oni.index.month])
autumn = np.array([month in [12] for month in oni.index.month])

winter = np.array([month in [12] for month in oni.index.month])
winter_p1 = np.array([month in [1, 2] for month in oni.index.month])


cp_m1 = np.array([year in elnino_cp - 1 for year in oni.index.year])
cp = np.array([year in elnino_cp for year in oni.index.year])
cp_p1 = np.array([year in elnino_cp + 1 for year in oni.index.year])

ep_m1 = np.array([year in elnino_ep - 1 for year in oni.index.year])
ep = np.array([year in elnino_ep for year in oni.index.year])
ep_p1 = np.array([year in elnino_ep + 1 for year in oni.index.year])


autumn_m1_cp = autumn & cp_m1
autumn_m1_ep = autumn & ep_m1
autumn_m1_nino = autumn_m1_cp | autumn_m1_ep

winter_m1_cp = (winter & cp_m1) | (winter_p1 & cp)
winter_m1_ep = (winter & ep_m1) | (winter_p1 & ep)
winter_m1_nino = winter_m1_cp | winter_m1_ep

spring_cp = (spring & cp)
spring_ep = (spring & ep)
spring_nino = spring_cp | spring_ep

summer_cp = (summer & cp)
summer_ep = (summer & ep)
summer_nino = summer_cp | summer_ep

autumn_cp = (autumn & cp)
autumn_ep = (autumn & ep)
autumn_nino = autumn_cp | autumn_ep

winter_cp = (winter & cp) | (winter_p1 & cp_p1)
winter_ep = (winter & ep) | (winter_p1 & ep_p1)
winter_nino = winter_cp | winter_ep

index_cp = (winter_cp)
index_ep = (winter_ep)


# =============================================================================
# Read data
# =============================================================================
taux = reader.read_netcdf('taux', dataset='NCEP', processed='anom')
taux = taux.sortby('lat', ascending=False)
tauy = reader.read_netcdf('tauy', dataset='NCEP', processed='anom')
tauy = taux.sortby('lat', ascending=False)

sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom')
sst = sst.sortby('lat', ascending=False)
ssh = reader.read_netcdf('zos', dataset='ORAS4', processed='anom')
ssh = ssh.sortby('lat', ascending=False)
olr =  - reader.read_netcdf('olr', dataset='NCAR', processed='anom')
olr = olr.sortby('lat', ascending=False)




taux_mean_cp = taux[index_cp,:,:].mean(dim='time', skipna=True)
taux_mean_ep = taux[index_ep,:,:].mean(dim='time', skipna=True)

tauy_mean_cp = tauy[index_cp,:,:].mean(dim='time', skipna=True)
tauy_mean_ep = tauy[index_ep,:,:].mean(dim='time', skipna=True)

sst_mean_cp = sst[index_cp,:,:].mean(dim='time', skipna=True)
sst_mean_ep = sst[index_ep,:,:].mean(dim='time', skipna=True)
ssh_mean_cp = ssh[index_cp,:,:].mean(dim='time', skipna=True)
ssh_mean_ep = ssh[index_ep,:,:].mean(dim='time', skipna=True)


olr_mean_cp = olr[index_cp,:,:].mean(dim='time', skipna=True)
olr_mean_ep = olr[index_ep,:,:].mean(dim='time', skipna=True)


#%% =============================================================================
# #Plots
# =============================================================================
levels_olr = levels = np.arange(-80.,90., 10)
levels_tau = np.round(np.arange(-100.,105., 5), decimals=1)
levels_ssh = np.round(np.arange(-1, 1.05, .05), decimals=2)
levels_sst = np.arange(-3, 3.25, 0.25)

plt.close("all")
lon2, lat2 = np.meshgrid(taux_mean_cp.lon, taux_mean_ep.lat)

# CP type
fig, axs = plt.subplots(2, 2, figsize=(12,4))




# SST and taux
m = Basemap(projection='merc',llcrnrlat=-30,urcrnrlat=30,\
            llcrnrlon=100,urcrnrlon=300,lat_ts=5,resolution='c',ax=axs[0,0])

x, y = m(lon2, lat2)

m.drawparallels(np.arange(-90., 120., 15.), labels=[1,0,0,0], color='grey')
m.drawmeridians(np.arange(0., 360., 30.), color='grey')
m.drawmapboundary(fill_color='white')
m.drawcoastlines()

ls = np.where(levels_ssh > 0, "-", "--")
ls[levels_ssh==0] = ':'
cs = m.contour(x, y, ssh_mean_cp, colors='black', levels=levels_ssh, linestyles=ls)

cs_sst = m.contourf(x, y, sst_mean_cp, cmap=plt.cm.seismic,levels=levels_sst, extend='both')

divider = make_axes_locatable(axs[0,0])
cax2 = divider.append_axes("right", size="8%", pad=0.1)
cax2.axis('off')


axs[0,0].text(9.2e5, 5.3e6, 'a', weight='bold', size=18,
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 7})






#OLR and taux
m = Basemap(projection='merc',llcrnrlat=-30,urcrnrlat=30,\
            llcrnrlon=100,urcrnrlon=300,lat_ts=5,resolution='c',ax=axs[1,0])

x, y = m(lon2, lat2)

m.drawparallels(np.arange(-90., 120., 15.), labels=[1,0,0,0], color='grey')
m.drawmeridians(np.arange(0., 360., 30.), labels=[0,0,0,1], color='grey')
m.drawmapboundary(fill_color='white')
m.drawcoastlines()

ls = np.where(levels_tau > 0, "-", "--")
ls[levels_tau==0] = ':'
cs = m.contour(x, y, taux_mean_cp, colors='black', levels=levels_tau, linestyles=ls)

cs_olr = m.contourf(x, y, olr_mean_cp, cmap=plt.cm.BrBG, levels=levels_olr, extend='both')
divider = make_axes_locatable(axs[1,0])
cax = divider.append_axes("right", size="8%", pad=0.1)
cax.axis('off')



axs[1,0].text(9.2e5, 5.3e6, 'b', weight='bold', size=18,
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 7})






# =============================================================================
#
# =============================================================================
# EP type
# SST and taux
m = Basemap(projection='merc',llcrnrlat=-30,urcrnrlat=30,\
            llcrnrlon=100,urcrnrlon=300,lat_ts=5,resolution='c',ax=axs[0,1])

x, y = m(lon2, lat2)

m.drawparallels(np.arange(-90., 120., 15.),color='grey')
m.drawmeridians(np.arange(0., 360., 30.),color='grey')
m.drawmapboundary(fill_color='white')
m.drawcoastlines()

ls = np.where(levels_ssh > 0, "-", "--")
ls[levels_ssh==0] = ':'
cs = m.contour(x, y, ssh_mean_ep, colors='black', levels=levels_ssh, linestyles=ls)

cs_sst = m.contourf(x, y, sst_mean_ep, cmap=plt.cm.seismic,levels=levels_sst, extend='both')

divider = make_axes_locatable(axs[0,1])
cax = divider.append_axes("right", size="8%", pad=0.1)
plt.colorbar(cs_sst, cax=cax, orientation='vertical', label=r'SSTA [K]')

axs[0,1].text(9.2e5, 5.3e6, 'c', weight='bold', size=18,
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 7})







#OLR and taux
m = Basemap(projection='merc',llcrnrlat=-30,urcrnrlat=30,\
            llcrnrlon=100,urcrnrlon=300,lat_ts=5,resolution='c',ax=axs[1,1])

x, y = m(lon2, lat2)

m.drawparallels(np.arange(-90., 120., 15.),color='grey')
m.drawmeridians(np.arange(0., 360., 30.),labels=[0,0,0,1],color='grey')
m.drawmapboundary(fill_color='white')
m.drawcoastlines()

ls =np.where(levels_tau > 0, "-", "--")
ls[levels_tau==0] = ':'
cs = m.contour(x, y, taux_mean_ep, colors='black', levels=levels_tau, linestyles=ls)

cs_olr = m.contourf(x, y, olr_mean_ep, cmap=plt.cm.BrBG, levels=levels_olr, extend='both')

divider = make_axes_locatable(axs[1,1])
cax = divider.append_axes("right", size="8%", pad=0.1)
plt.colorbar(cs_olr, cax=cax, label=r'- OLRA [W/m$^2$]')

axs[1,1].text(9.2e5, 5.3e6, 'd', weight='bold', size=18,
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 7})

plt.tight_layout()

if save:
    plt.savefig(join(plotdir, 'composite.pdf'))
    plt.savefig(join(plotdir, 'composite.jpg'), dpi=360)