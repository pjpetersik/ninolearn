import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from mpl_toolkits.basemap import Basemap
from os.path import join

from ninolearn.IO.read_post import data_reader
from ninolearn.utils import scale
from ninolearn.private import plotdir


plt.close("all")
# =============================================================================
# Data
# =============================================================================
reader = data_reader(startdate='1980-01', enddate='2018-12', lon_min=100,lon_max=300)

pca_dechca = reader.read_statistic('pca', variable='dec_hca', dataset='NODC', processed='anom')
pca_decsst = reader.read_statistic('pca', variable='dec_sst', dataset='ERSSTv5', processed='anom')


olr = reader.read_netcdf('olr', dataset='NCAR', processed='anom')
olr = olr.rolling(time=60, center=False).mean()
olr = olr.sortby('lat', ascending=False)

taux = reader.read_netcdf('taux', dataset='NCEP', processed='anom')
taux = taux.sortby('lat', ascending=False)

sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom')
sst = sst.sortby('lat', ascending=False)
# =============================================================================
# Model
# =============================================================================

X = np.stack((-pca_decsst['pca1'].loc['1985-01':],),axis=1)
y = olr.loc['1985-01':]

reg = linear_model.LinearRegression(fit_intercept=True)
flat_y = y.values.reshape(y.shape[0],-1)

reg.fit(X, flat_y)

pred = reg.predict(X)
score = r2_score(flat_y, pred, multioutput='raw_values')

score2 = np.zeros_like(score)
p = np.zeros_like(score)

for i in range(score2.shape[0]):
    score2[i], p[i] = pearsonr(pred[:,i], flat_y[:,i])


score_map = score.reshape((y.shape[1],y.shape[2]))
score2_map = score2.reshape((y.shape[1],y.shape[2]))
p_map = p.reshape((y.shape[1], y.shape[2]))

coef = reg.coef_[:,0].reshape((y.shape[1],y.shape[2])) * X[:,0].std()

# =============================================================================
# Plot
# =============================================================================
plt.close("all")
plt.plot(scale(pca_dechca['pca1']))
plt.plot(-scale(pca_decsst['pca1']))

# Setup
vmax = 10
levels = np.linspace(-vmax, vmax, 21, endpoint = True)
levels_r2 = np.linspace(0, 1, 21, endpoint = True)


# Generate the base plot
lon2, lat2 = np.meshgrid(y.lon, y.lat)
fig, axs = plt.subplots(1, 1, figsize=(9,5))

m = Basemap(projection='merc',llcrnrlat=-30,urcrnrlat=30,\
        llcrnrlon=100,urcrnrlon=300,lat_ts=5,resolution='c', ax=axs)

x, y = m(lon2, lat2)

m.drawparallels(np.arange(-90., 120., 15.), labels=[1,0,0,0], color='grey')

m.drawmeridians(np.arange(0., 360., 30.),  labels=[0,0,0,1], color='grey')



m.drawmapboundary(fill_color='white')
m.drawcoastlines()

cs_olr = m.contour(x, y, coef, vmin=-vmax, vmax=vmax, levels=levels, cmap=plt.cm.seismic)
cs_r2 =  m.contourf(x, y, score2_map**2, vmin=0.0,vmax=1, levels = levels_r2, cmap=plt.cm.Greens, extend='max')
cs_p = m.contourf(x, y, p_map, levels=[0, 0.001], hatches = ['//'], alpha=0)

# Color bar
fig.subplots_adjust(right=0.7)
cbar_ax1 = fig.add_axes([0.75, 0.15, 0.02, 0.7])
cbar_ax2 = fig.add_axes([0.85, 0.15, 0.02, 0.7])

fig.colorbar(cs_olr, cax= cbar_ax1, label=r'-OLR[W$\,$m$^{-2}$]')
fig.colorbar(cs_r2, cax = cbar_ax2, label=r'$r^2$')

#plt.savefig(join(plotdir, 'dec_olr_regression.pdf'))
