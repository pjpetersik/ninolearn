import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from mpl_toolkits.basemap import Basemap
from os.path import join
import pandas as pd

from ninolearn.IO.read_post import data_reader
from ninolearn.utils import scale
from ninolearn.private import plotdir
from ninolearn.postprocess.pca import pca
from composites_CP_EP import winter_nino


plt.close("all")



reader = data_reader(startdate='1955-01', enddate='2018-12',lon_min=100, lon_max=300)
sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom')
sst_decadel = sst.rolling(time=60, center=False).mean()
sst_decadel.attrs = sst.attrs.copy()
sst_decadel.name = f'dec_{sst.name}'

pca_sst = pca(n_components=6)
pca_sst.set_eof_array(sst_decadel)
pca_sst.compute_pca()

PC = - pca_sst.pc_projection(eof=1) / 10
ds_PC = pd.Series(PC, index=pd.date_range(start='1955-01-01', end='2018-12-01', freq='MS'))



exp_var =  np.round(pca_sst.explained_variance_ratio_[0] * 100, decimals=1)
fig, axs = plt.subplots(2, 1, figsize=(7,5))

axs[1].plot(pd.date_range(start='1955-01-01', end='2018-12-01', freq='MS'), PC)
axs[1].set_xlim('1960-01', '2018-12')
axs[1].set_ylim(-2, 1.5)
axs[1].hlines(0,'1960-01', '2018-12', linestyle='--')
axs[1].set_xlabel('Time [Year]')
axs[1].set_ylabel('Amplitude')
axs[1].text('1962-01', 0.75, 'b' , size=18, weight='bold', bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 7})

axs[1].text('2010-01',0.9, f'{exp_var}%',size=16)



# Generate the base plot
lon2, lat2 = np.meshgrid(pca_sst.lon, pca_sst.lat)

m = Basemap(projection='merc',llcrnrlat=-30,urcrnrlat=30,\
        llcrnrlon=100,urcrnrlon=300,lat_ts=5,resolution='c', ax=axs[0])

x, y = m(lon2, lat2)

m.drawparallels(np.arange(-90., 120., 15.), labels=[1,0,0,0], color='grey')

m.drawmeridians(np.arange(0., 360., 30.),  labels=[0,0,0,1], color='grey')


m.drawmapboundary(fill_color='white')
m.drawcoastlines()

vmax = 1.
levels = np.linspace(-vmax, vmax, 21, endpoint = True)
cs = m.contourf(x, y, -  10 * pca_sst.component_map_(eof=1), levels=levels, cmap=plt.cm.seismic, vmin=-vmax, vmax=vmax, extend='both')
m.colorbar(cs,  label=r'T [K]')
axs[0].text(9.2e5, 5.3e6, 'a', weight='bold', size=18,
        bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 7})


plt.savefig(join(plotdir, 'dec_sst_eof.pdf'))
plt.savefig(join(plotdir, 'dec_sst_eof.jpg'), dpi = 360)
#%%
# =============================================================================
# Data
# =============================================================================
reader = data_reader(startdate='1974-06', enddate='2018-12',lon_min=100, lon_max=300)

olr = reader.read_netcdf('olr', dataset='NCAR', processed='anom')
olr = olr.rolling(time=60, center=False).mean()
olr = olr.sortby('lat', ascending=False)

taux = reader.read_netcdf('taux', dataset='NCEP', processed='anom')
taux = taux.rolling(time=60, center=False).mean()
taux = taux.sortby('lat', ascending=False)


# =============================================================================
# Model
# =============================================================================




def regression_map(X, y):
    y_m = np.ma.masked_invalid(y)
    flat_y = y_m.reshape(y_m.shape[0],-1)
    flat_y_comp = np.ma.compress_rowcols(flat_y, axis=1)

    reg = linear_model.LinearRegression(fit_intercept=True)
    reg.fit(X, flat_y_comp)

    pred = reg.predict(X)

    score_m = np.ma.masked_array(np.zeros_like(flat_y[1]), flat_y.mask[1])
    p_m = np.ma.masked_array(np.zeros_like(flat_y[1]), flat_y.mask[1])

    indeces = np.argwhere(score_m.mask==False)[:,0]

    for i in range(pred.shape[1]):
        j = indeces[i]
        score_m[j], p_m[j] = pearsonr(pred[:,i], flat_y_comp[:,i])

    r, p = score_m.filled(np.nan), p_m.filled(np.nan)

    r_map = r.reshape((y.shape[1],y.shape[2]))
    p_map = p.reshape((y.shape[1], y.shape[2]))

    coef_m = np.ma.masked_array(np.zeros_like(flat_y[1]), flat_y.mask[1])
    coef_m[indeces] = reg.coef_[:,0] * X[:,0].std()

    coef = coef_m.filled(np.nan)
    coef_map = coef.reshape((y.shape[1],y.shape[2]))

    return r_map, p_map, coef_map


y_tau = taux.loc['1985-01':]
X = np.stack((ds_PC.loc['1985-01':],),axis=1)

r_tau, p_tau, coef_tau = regression_map(X, y_tau)

y_olr = olr.loc['1985-01':]
r_olr, p_olr, coef_olr = regression_map(X, y_olr)


# =============================================================================
# Plot
# =============================================================================
# Setup
vmax = 10
levels_tau = levels_olr = np.linspace(-vmax, vmax, 21, endpoint = True)
levels_r2 = np.linspace(0, 1, 21, endpoint = True)


# Generate the base plot
fig, axs = plt.subplots(2, 1, figsize=(8,4))
lon2, lat2 = np.meshgrid(y_tau.lon, y_tau.lat)
m = []



for i in range(2):
    m.append( Basemap(projection='merc',llcrnrlat=-30,urcrnrlat=30,\
            llcrnrlon=100,urcrnrlon=300,lat_ts=5,resolution='c', ax=axs[i]))
    x, y = m[i](lon2, lat2)

    m[i].drawparallels(np.arange(-90., 120., 15.), labels=[1,0,0,0], color='grey')
    m[i].drawmeridians(np.arange(0., 360., 30.),  labels=[0,0,0,1], color='grey')
    m[i].drawmapboundary(fill_color='white')
    m[i].drawcoastlines()

axs[0].text(9.2e5, 5.3e6, 'a', weight='bold', size=18,
    bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 7})
axs[1].text(9.2e5, 5.3e6, 'b', weight='bold', size=18,
    bbox={'facecolor': 'white', 'alpha': 0.9, 'pad': 7})


cs_olr = m[0].contour(x, y, coef_olr, vmin=-vmax, vmax=vmax, levels=levels_olr, cmap=plt.cm.seismic, extend='both')
cs_r2 =  m[0].contourf(x, y, r_olr**2, vmin=0.0,vmax=1, levels = levels_r2, cmap=plt.cm.Greens, extend='max')
cs_p = m[0].contourf(x, y, p_olr, levels=[0, 0.001], hatches = ['//'], alpha=0)

cs_olr = m[1].contour(x, y, coef_tau, vmin=-vmax, vmax=vmax, levels=levels_tau, cmap=plt.cm.seismic, extend='both')
cs_r2 =  m[1].contourf(x, y, r_tau**2, vmin=0.0,vmax=1, levels = levels_r2, cmap=plt.cm.Greens, extend='max')
cs_p = m[1].contourf(x, y, p_tau, levels=[0, 0.001], hatches = ['//'], alpha=0)


# Color bar
fig.subplots_adjust(right=0.75)
cbar_ax1 = fig.add_axes([0.75, 0.1, 0.04, 0.8])
cbar_ax2 = fig.add_axes([0.87, 0.1, 0.04, 0.8])

fig.colorbar(cs_olr, cax= cbar_ax1, label=r'OLR [W$\,$m$^{-2}$]  /  $\hat\tau_x$ [m$^2\,$s$^{-2}$]')
fig.colorbar(cs_r2, cax = cbar_ax2, label=r'$r^2$')




plt.savefig(join(plotdir, 'dec_olr_regression.pdf'))
plt.savefig(join(plotdir, 'dec_olr_regression.jpg'), dpi = 360)