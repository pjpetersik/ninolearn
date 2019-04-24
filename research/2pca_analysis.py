from ninolearn.postprocess.pca import pca

# =============================================================================
# =============================================================================
# # Compute PCA
# =============================================================================
# =============================================================================
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

#%% =============================================================================
# GFDL data
# =============================================================================
pca_sat_gfdl = pca(n_components=6)
pca_sat_gfdl.load_data('tas', 'GFDL-CM3', processed="anom",
                  startyear=1700, endyear=2199, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30)

pca_sat_gfdl.compute_pca()
pca_sat_gfdl.save()

pca_sat_gfdl.plot_eof()