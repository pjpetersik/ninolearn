from ninolearn.IO.read_post import data_reader
from ninolearn.pathes import postdir

from os.path import join
import pandas as pd

# =============================================================================
# Nino3.4
# =============================================================================
reader = data_reader(startdate='1700-01-01', enddate='2199-12-01',
                     lon_min=190, lon_max=240,
                     lat_min=-5, lat_max=5)

sst = reader.read_netcdf('tos', dataset='GFDL-CM3', processed='')

nino34 = sst.mean(dim= ['lat','lon'])

nino34_anom = nino34 - nino34.mean()

pd_nino34_anom = nino34_anom.to_pandas()

df = pd.DataFrame(data=pd_nino34_anom.values,index=pd_nino34_anom.index, columns=['anom'])

df.to_csv(join(postdir, 'nino3.4M_gfdl.csv'))

# =============================================================================
# IOD
# =============================================================================

reader_west = data_reader(startdate='1700-01-01', enddate='2199-12-01',
                     lon_min=50, lon_max=70,
                     lat_min=-10, lat_max=10)

sst_west = reader_west.read_netcdf('tos', dataset='GFDL-CM3', processed='')
sst_west = sst_west.mean(dim= ['lat','lon'])
sst_west_anom = sst_west.groupby('time.month') - sst_west.groupby(f'time.month').mean(dim="time")

reader_southeast = data_reader(startdate='1700-01-01', enddate='2199-12-01',
                     lon_min=90, lon_max=110,
                     lat_min=-10, lat_max=0)

sst_southeast = reader_southeast.read_netcdf('tos', dataset='GFDL-CM3', processed='')
sst_southeast = sst_southeast.mean(dim= ['lat','lon'])

sst_southeast_anom = sst_southeast.groupby('time.month') - sst_southeast.groupby(f'time.month').mean(dim="time")

iod = sst_west_anom - sst_southeast_anom

pd_iod = iod.to_pandas()

df_iod = pd.DataFrame(data=pd_iod.values,index=pd_iod.index, columns=['anom'])

df_iod.to_csv(join(postdir, 'iod_gfdl.csv'))



