from ninolearn.postprocess.pca import pca
from ninolearn.IO.read_post import data_reader
import matplotlib.pyplot as plt

plt.close("all")

# =============================================================================
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


