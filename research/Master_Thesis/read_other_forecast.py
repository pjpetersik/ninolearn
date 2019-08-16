from os.path import join
import pandas as pd
import xarray as xr
import numpy as np
from ninolearn.pathes import rawdir, postdir
from datetime import datetime

data = pd.read_csv(join(rawdir, "other_forecasts.csv"), error_bad_lines=False,
                   header=None, names=['row'], delimiter=';')

n_rows = len(data)

# the first target season is assumed not to change
first_target_season = '2002-04-01'

for i in reversed(range(n_rows)):
    row_str = data.row[i]
    if row_str[:8] == "Forecast":
        last_issued =datetime.strptime(row_str[16:], '%b %Y')
        break

last_target_season = last_issued + pd.tseries.offsets.MonthBegin(10)

target_season = pd.date_range(start=first_target_season, end=last_target_season, freq='MS')

n_timesteps = len(target_season)

lead_time = np.arange(0, 9)
n_lead = len(lead_time)
#%%
dummy = np.zeros((n_timesteps, n_lead))
dummy[:,:] = np.nan

raw_UBC_NNET = dummy.copy()
raw_CFS = dummy.copy()
raw_ECMWF = dummy.copy()
raw_CPC_MRKOV = dummy.copy()
raw_CPC_CA = dummy.copy()
raw_CPC_CCA = dummy.copy()

raw_UCLA_TCD = dummy.copy()
raw_JMA = dummy.copy()
raw_NASA_GMAO = dummy.copy()
raw_SCRIPPS = dummy.copy()
raw_KMA_SNU = dummy.copy()

j = 0

save_data = {}

for i in range(n_rows):
    try:
        row_str = data.row[i]

        if row_str[:3]=='end':
            j+=1

        # check if it is a number otherwise except
        test = int(row_str[0:4])

        model_name = row_str[40:52].strip()

        if model_name not in save_data.keys() and model_name!='':
            save_data[model_name] =  dummy.copy()

        for k in range(9):
            save_data[model_name][j+k, k] = int(row_str[0+k*4: 4+k*4])/100


        # Dynamical
        if model_name == "NCEP CFSv2" or model_name.strip() == "NCEP CFS":
            for k in range(9):
                raw_CFS[j+k, k] = int(row_str[0+k*4: 4+k*4])


        if model_name == "JMA":
            for k in range(9):
                raw_JMA[j+k, k] = int(row_str[0+k*4: 4+k*4])

        if model_name == "NASA GMAO":
            for k in range(9):
                raw_NASA_GMAO[j+k, k] = int(row_str[0+k*4: 4+k*4])

        if model_name == "NASA GMAO":
            for k in range(9):
                raw_NASA_GMAO[j+k, k] = int(row_str[0+k*4: 4+k*4])

        if model_name == "ECMWF":
            for k in range(9):
                raw_ECMWF[j+k, k] = int(row_str[0+k*4: 4+k*4])

        if model_name == "SCRIPPS":
            for k in range(9):
                raw_SCRIPPS[j+k, k] = int(row_str[0+k*4: 4+k*4])

        if model_name == "KMA SNU":
            for k in range(9):
                raw_KMA_SNU[j+k, k] = int(row_str[0+k*4: 4+k*4])

        # statistical
        if model_name == "CPC CA":
            for k in range(9):
                raw_CPC_CA[j+k, k] = int(row_str[0+k*4: 4+k*4])

        if model_name == "CPC CCA":
            for k in range(9):
                raw_CPC_CCA[j+k, k] = int(row_str[0+k*4: 4+k*4])


        if model_name == "UCLA-TCD":
            for k in range(9):
                raw_UCLA_TCD[j+k, k] = int(row_str[0+k*4: 4+k*4])


        if model_name == "UBC NNET":
            for k in range(9):
                raw_UBC_NNET[j+k, k] = int(row_str[0+k*4: 4+k*4])


        # CPC MRKOV has forecasts for all times
        # for issued forecasts from 2002-02 till 2019-07
        # therefore it is last and with the j+=1 in this condition
        if model_name == "CPC MRKOV":
            for k in range(9):
                raw_CPC_MRKOV[j+k, k] = int(row_str[0+k*4: 4+k*4])


    except ValueError:
        pass

ds = xr.Dataset({'UBC NNET': (['target_season', 'lead'],  raw_UBC_NNET),
                 'NCEP CFS': (['target_season', 'lead'],  raw_CFS),
                 'ECMWF': (['target_season', 'lead'],  raw_ECMWF),
                 'CPC_MRKOV': (['target_season', 'lead'],  raw_CPC_MRKOV),
                 'CPC_CA': (['target_season', 'lead'],  raw_CPC_CA),
                 'CPC_CCA': (['target_season', 'lead'],  raw_CPC_CCA),
                 'UCLA-TCD': (['target_season', 'lead'],  raw_UCLA_TCD),
                 'JMA': (['target_season', 'lead'],  raw_JMA),
                 'NASA GMAO': (['target_season', 'lead'],  raw_NASA_GMAO),
                 'SCRIPPS': (['target_season', 'lead'],  raw_SCRIPPS),
                 'KMA SNU': (['target_season', 'lead'],  raw_KMA_SNU),
                 },
                 coords={'target_season': target_season,
                         'lead': lead_time
                         }
                )

save_dict = {}
for key in save_data.keys():
    save_dict[key] = (['target_season', 'lead'], save_data[key])

ds2 = xr.Dataset(save_dict,
                 coords={'target_season': target_season,
                         'lead': lead_time
                         }
                )


ds=ds.where(ds!=-999)
ds = ds/100
ds.to_netcdf(join(postdir, f'other_forecasts.nc'))