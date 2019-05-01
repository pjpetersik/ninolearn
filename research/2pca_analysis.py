from ninolearn.postprocess.pca import pca

import matplotlib.pyplot as plt
plt.close("all")
# =============================================================================
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

# sst
pca_sst = pca(n_components=6)
pca_sst.load_data('sst', 'ERSSTv5', processed="anom",
                  startyear=1948, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)
pca_sst.compute_pca()
pca_sst.save()


# ssh
pca_ssh = pca(n_components=6)
pca_ssh.load_data('sshg', 'GODAS', processed="anom",
                  startyear=1980, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)
pca_ssh.compute_pca()
pca_ssh.save()
pca_ssh.plot_eof()

# =============================================================================
# #%% Indic
# =============================================================================
pca_sst_indic = pca(n_components=6)
pca_sst_indic.load_data('sst', 'ERSSTv5', processed="anom",
                  startyear=1948, endyear=2018, lon_min=50, lon_max=110,
                  lat_min=-10, lat_max=10)
pca_sst_indic.compute_pca()
pca_sst_indic.save(extension='_iod')
pca_sst_indic.plot_eof()


#%% =============================================================================
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

