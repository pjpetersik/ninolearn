from mpl_toolkits.basemap import Basemap
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from ninolearn.IO.read_post import data_reader
# =============================================================================
# Read
# =============================================================================
reader = data_reader(startdate='1980-01', enddate='2018-11', lon_min=100, lon_max=300)
iod = reader.read_csv('iod')
nino = reader.read_csv('nino3M')

taux = reader.read_netcdf('taux', dataset='NCEP', processed='anom')
taux = taux.sortby('lat', ascending=False)

sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom')
sst = sst.sortby('lat', ascending=False)

olr = - reader.read_netcdf('olr', dataset='NCAR', processed='anom')
olr = olr.sortby('lat', ascending=False)

# =============================================================================
# Regression analysis
# =============================================================================
SSTA_gradient = np.nanmean(np.gradient(sst.loc[dict(lat=0, lon=slice(120, 280))], axis=0), axis=1)
OLR_mean = olr.loc[dict(lat=slice(5, -5), lon=slice(150, 210))].mean(dim='lon').mean(dim='lat')

X = np.stack((nino, OLR_mean), axis=1)
reg = linear_model.LinearRegression(fit_intercept=False)

taux_flat = taux.values.reshape(taux.shape[0],-1)

reg.fit(X, taux_flat)

pred = reg.predict(X)
score = r2_score(taux_flat, pred, multioutput='raw_values')

score2 = np.zeros_like(score)
p = np.zeros_like(score)

for i in range(score2.shape[0]):
    score2[i], p[i] = pearsonr(pred[:,i], taux_flat[:,i])

score_map = score.reshape((taux.shape[1],taux.shape[2]))
score2_map = score2.reshape((taux.shape[1],taux.shape[2]))
p_map = p.reshape((taux.shape[1],taux.shape[2]))


coef_sst = reg.coef_[:,0].reshape((taux.shape[1],taux.shape[2]))*nino.std()
coef_olr = reg.coef_[:,1].reshape((taux.shape[1],taux.shape[2]))*OLR_mean.values.std()
#coef_iod = reg.coef_[:,2].reshape((taux.shape[1],taux.shape[2]))
# =============================================================================
# Plot
# =============================================================================
plt.close("all")
lon2, lat2 = np.meshgrid(taux.lon, taux.lat)

fig, axs = plt.subplots(2, 1, figsize=(9,5))
# SST gradient regression
m = Basemap(projection='merc',llcrnrlat=-30,urcrnrlat=30,\
            llcrnrlon=100,urcrnrlon=300,lat_ts=5,resolution='c', ax=axs[0])

x, y = m(lon2, lat2)

m.drawparallels(np.arange(-90., 120., 15.), labels=[1,0,0,0], color='grey')
m.drawmeridians(np.arange(0., 360., 30.),  labels=[0,0,0,1], color='grey')
m.drawmapboundary(fill_color='white')
m.drawcoastlines()

vmax = 10

levels = np.linspace(-vmax, vmax, 21, endpoint = True)

cs1 = m.contour(x, y, coef_sst, vmin=-vmax, vmax=vmax, levels=levels, cmap=plt.cm.seismic)
cs_r2 =  m.contourf(x, y, score2_map**2, vmin=0.0,vmax=0.8,cmap=plt.cm.Greens)
cs_p = m.contourf(x, y, p_map, levels=[0, 0.01], hatches = ['//'], alpha=0)
#plt.colorbar(cs1, ax=axs[0])


# OLR regression
m = Basemap(projection='merc',llcrnrlat=-30,urcrnrlat=30,\
            llcrnrlon=100,urcrnrlon=300,lat_ts=5,resolution='c', ax=axs[1])

x, y = m(lon2, lat2)

m.drawparallels(np.arange(-90., 120., 15.), labels=[1,0,0,0], color='grey')
m.drawmeridians(np.arange(0., 360., 30.),  labels=[0,0,0,1], color='grey')
m.drawmapboundary(fill_color='white')
m.drawcoastlines()


cs = m.contour(x, y, coef_olr, vmin=-vmax, vmax=vmax, levels=levels, cmap=plt.cm.seismic)
cs_r2 =  m.contourf(x, y, score2_map**2, vmin=0.0,vmax=0.8, cmap=plt.cm.Greens, extend='max')
cs_p = m.contourf(x, y, p_map, levels=[0, 0.01], hatches = ['//'], alpha=0)

fig.subplots_adjust(right=0.7)
cbar_ax1 = fig.add_axes([0.75, 0.15, 0.02, 0.7])
cbar_ax2 = fig.add_axes([0.85, 0.15, 0.02, 0.7])

fig.colorbar(cs, cax= cbar_ax1, label=r'$\tau_x$[m$^2$s$^{-2}$]')

fig.colorbar(cs_r2, cax= cbar_ax2, label=r'$r^2$')


#plt.colorbar(cs, ax=axs[1])
#plt.colorbar(cs_r2, ax=axs[1])



# IOD regression
#m = Basemap(projection='merc',llcrnrlat=-30,urcrnrlat=30,\
#            llcrnrlon=30,urcrnrlon=300,lat_ts=5,resolution='c', ax=axs[2])
#
#x, y = m(lon2, lat2)
#
#m.drawparallels(np.arange(-90., 120., 15.), labels=[1,0,0,0], color='grey')
#m.drawmeridians(np.arange(0., 360., 30.),  labels=[0,0,0,1], color='grey')
#m.drawmapboundary(fill_color='white')
#m.drawcoastlines()
#
#vmax = np.max(coef_iod)
#levels = np.arange(-10,12,2)
#cs = m.contour(x, y, coef_iod, vmin=-vmax, vmax=vmax, levels=levels,cmap=plt.cm.seismic)
#plt.colorbar(cs)