from ninolearn.postprocess.pca import pca
from ninolearn.IO.read_post import data_reader
import matplotlib.pyplot as plt
import numpy as np
plt.close("all")
#%% =============================================================================
# =============================================================================
# # Reanalysis
# =============================================================================
# =============================================================================

# =============================================================================
# # Pacific
# =============================================================================
#sat
pca_sat = pca(n_components=6)
pca_sat.load_data('air', 'NCEP', processed="anom",
                  startyear=1948, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)
pca_sat.compute_pca()
pca_sat.save()

#uwnd
pca_uwnd = pca(n_components=6)
pca_uwnd.load_data('uwnd', 'NCEP', processed="anom",
                  startyear=1948, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)
pca_uwnd.compute_pca()
pca_uwnd.save()
pca_uwnd.plot_eof()

# sst
pca_sst = pca(n_components=6)
pca_sst.load_data('sst', 'ERSSTv5', processed="anom",
                  startyear=1948, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)
pca_sst.compute_pca()
pca_sst.save()
pca_sst.plot_eof()


#%% ssh
pca_ssh = pca(n_components=6)
pca_ssh.load_data('sshg', 'GODAS', processed="anom",
                  startyear=1980, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)
pca_ssh.compute_pca()
pca_ssh.save()

#taux
pca_taux = pca(n_components=6)
pca_taux.load_data('taux', 'NCEP', processed="anom",
                  startyear=1948, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)
pca_taux.compute_pca()
pca_taux.save()
pca_taux.plot_eof()



#%% =============================================================================
# Decadel PCAs
# =============================================================================
# TODO: The naming of the saved files is not perfect. better would be a automatic
# filename generation that aligns with the  naming convention

reader = data_reader(startdate='1955-02', enddate='2018-12', lon_min=120, lon_max=300)
hca = reader.read_netcdf('hca', dataset='NODC', processed='anom')

hca_decadel = hca.rolling(time=60, center=False).mean()
hca_decadel.attrs = hca.attrs.copy()
hca_decadel.name = f'dec_{hca.name}'

pca_hca_decadel = pca(n_components=6)

pca_hca_decadel.set_eof_array(hca_decadel)
pca_hca_decadel.compute_pca()
pca_hca_decadel.plot_eof()
pca_hca_decadel.save(extension='.csv', filename='dec_hca_NODC_anom')



reader = data_reader(startdate='1955-01', enddate='2018-12',lon_min=120, lon_max=300)
sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom')

sst_decadel = sst.rolling(time=60, center=False).mean()
sst_decadel.attrs = sst.attrs.copy()
sst_decadel.name = f'dec_{sst.name}'

pca_sst_decadel = pca(n_components=6)

pca_sst_decadel.set_eof_array(sst_decadel)
pca_sst_decadel.compute_pca()
pca_sst_decadel.plot_eof()
pca_sst_decadel.save(extension='.csv', filename='dec_sst_ERSSTv5_anom')


#%%olr
pca_olr = pca(n_components=6)
pca_olr.load_data('olr', 'NCAR', processed="anom",
                  startyear=1975, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)
pca_olr.compute_pca()
pca_olr.save()
pca_olr.plot_eof()

#%%OLR long
reader = data_reader(startdate='1975-01', enddate='2018-12',lon_min=120, lon_max=300)

olr = reader.read_netcdf('olr', dataset='NCAR', processed='anom')

olr_decadel = olr.rolling(time=60, center=False).mean()
olr_decadel.attrs = olr.attrs.copy()
olr_decadel.name = f'dec_{olr.name}'

pca_olr_decadel = pca(n_components=6)

pca_olr_decadel.set_eof_array(olr_decadel)
pca_olr_decadel.compute_pca()
pca_olr_decadel.plot_eof()
pca_olr_decadel.save(extension='.csv', filename='dec_sst_ERSSTv5_anom')


"""
ARCHIEVED


pca_sat = pca(n_components=6)
pca_sat.load_data('air_daily', 'NCEP', processed="anom",
                  startyear=1948, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)
pca_sat.compute_pca()
pca_sat.save()

pca_sat = pca(n_components=6)
pca_sat.load_data('uwnd', 'NCEP', processed="anom",
                  startyear=1948, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)
pca_sat.compute_pca()
pca_sat.save()

pca_sat = pca(n_components=6)
pca_sat.load_data('vwnd', 'NCEP', processed="anom",
                  startyear=1948, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)
pca_sat.compute_pca()
pca_sat.save()

# =============================================================================
# #%% Indic
# =============================================================================
pca_sst_indic = pca(n_components=6)
pca_sst_indic.load_data('sst', 'ERSSTv5', processed="anom",
                  startyear=1948, endyear=2018, lon_min=50, lon_max=110,
                  lat_min=-10, lat_max=10)
pca_sst_indic.compute_pca()
pca_sst_indic.save(extension='_iod')

# =============================================================================
# =============================================================================
# # GFDL data
# =============================================================================
# =============================================================================
pca_sat_gfdl = pca(n_components=6)
pca_sat_gfdl.load_data('tas', 'GFDL-CM3', processed="anom",
                  startyear=1700, endyear=2199, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)

pca_sat_gfdl.compute_pca()
pca_sat_gfdl.save()

pca_sst_gfdl = pca(n_components=6)
pca_sst_gfdl.load_data('tos', 'GFDL-CM3', processed="anom",
                  startyear=1700, endyear=2199, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)

pca_sst_gfdl.compute_pca()
pca_sst_gfdl.plot_eof()
pca_sst_gfdl.save()
"""