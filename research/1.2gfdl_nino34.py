from ninolearn.IO.read_post import data_reader
from ninolearn.pathes import postdir

from os.path import join
import pandas as pd

reader = data_reader(startdate='1700-01-01', enddate='2199-12-01',
                     lon_min=190, lon_max=240,
                     lat_min=-5, lat_max=5)

sat = reader.read_netcdf('tos', dataset='GFDL-CM3', processed='')

nino34 = sat.mean(dim= ['lat','lon'])

nino34_anom = nino34 - nino34.mean()

pd_nino34_anom = nino34_anom.to_pandas()

df = pd.DataFrame(data=pd_nino34_anom.values,index=pd_nino34_anom.index, columns=['anom'])

df.to_csv(join(postdir, 'nino3.4M_gfdl.csv'))
