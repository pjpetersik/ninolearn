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

X = np.stack((nino, OLR_mean, iod), axis=1)
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


coef_sst = reg.coef_[:,0].reshape((taux.shape[1],taux.shape[2])) * X[:,0].std()
coef_olr = reg.coef_[:,1].reshape((taux.shape[1],taux.shape[2])) * X[:,1].std()
coef_iod = reg.coef_[:,2].reshape((taux.shape[1],taux.shape[2])) * X[:,2].std()
# =============================================================================
# Plot
# =============================================================================
plt.close("all")

# Setup
vmax = 10
levels = np.linspace(-vmax, vmax, 21, endpoint = True)
levels_r2 = np.linspace(0, 0.8, 21, endpoint = True)


# Generate the base plot
lon2, lat2 = np.meshgrid(taux.lon, taux.lat)
fig, axs = plt.subplots(3, 1, figsize=(9,5))
m = []
for i in range(3):
    m.append(Basemap(projection='merc',llcrnrlat=-30,urcrnrlat=30,\
            llcrnrlon=100,urcrnrlon=300,lat_ts=5,resolution='c', ax=axs[i]))

    x, y = m[i](lon2, lat2)

    m[i].drawparallels(np.arange(-90., 120., 15.), labels=[1,0,0,0], color='grey')
    if i == 2:
        m[i].drawmeridians(np.arange(0., 360., 30.),  labels=[0,0,0,1], color='grey')
    else:
        m[i].drawmeridians(np.arange(0., 360., 30.),  color='grey')


    m[i].drawmapboundary(fill_color='white')
    m[i].drawcoastlines()

    cs_r2 =  m[i].contourf(x, y, score2_map**2, vmin=0.0,vmax=0.8, levels = levels_r2, cmap=plt.cm.Greens, extend='max')
    cs_p = m[i].contourf(x, y, p_map, levels=[0, 0.01], hatches = ['//'], alpha=0)


# Overlay the base plot with
cs_sst = m[0].contour(x, y, coef_sst, vmin=-vmax, vmax=vmax, levels=levels, cmap=plt.cm.seismic)
cs_olr = m[1].contour(x, y, coef_olr, vmin=-vmax, vmax=vmax, levels=levels, cmap=plt.cm.seismic)
cs_iod = m[2].contour(x, y, coef_iod, vmin=-vmax, vmax=vmax, levels=levels, cmap=plt.cm.seismic)






# Color bar
fig.subplots_adjust(right=0.7)
cbar_ax1 = fig.add_axes([0.75, 0.15, 0.02, 0.7])
cbar_ax2 = fig.add_axes([0.85, 0.15, 0.02, 0.7])

fig.colorbar(cs_sst, cax= cbar_ax1, label=r'$\tau_x$[m$^2$s$^{-2}$]')
fig.colorbar(cs_r2, cax = cbar_ax2, label=r'$r^2$')



