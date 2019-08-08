from os.path import join
import pandas as pd
import xarray as xr
import numpy as np
from ninolearn.pathes import rawdir, postdir


data = pd.read_csv(join(rawdir, "other_forecasts.csv"), error_bad_lines=False)

dummy = np.zeros((211, 9))
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
for i in range(len(data)):
    try:
        # check if it is a number otherwise except
        test = int(data.Forecast[i][0:4])
        model_name = data.Forecast[i][40:52]

        # 103
        if j<203:
            # Dynamical
            if model_name.strip() == "NCEP CFSv2" or model_name.strip() == "NCEP CFS":
                for k in range(9):
                    raw_CFS[j+k, k] = int(data.Forecast[i][0+k*4: 4+k*4])

            if model_name.strip() == "JMA":
                for k in range(9):
                    raw_JMA[j+k, k] = int(data.Forecast[i][0+k*4: 4+k*4])

            if model_name.strip() == "NASA GMAO":
                for k in range(9):
                    raw_NASA_GMAO[j+k, k] = int(data.Forecast[i][0+k*4: 4+k*4])

            if model_name.strip() == "NASA GMAO":
                for k in range(9):
                    raw_NASA_GMAO[j+k, k] = int(data.Forecast[i][0+k*4: 4+k*4])

            if model_name.strip() == "ECMWF":
                for k in range(9):
                    raw_ECMWF[j+k, k] = int(data.Forecast[i][0+k*4: 4+k*4])

            if model_name.strip() == "SCRIPPS":
                for k in range(9):
                    raw_SCRIPPS[j+k, k] = int(data.Forecast[i][0+k*4: 4+k*4])


            if model_name.strip() == "KMA SNU":
                for k in range(9):
                    raw_KMA_SNU[j+k, k] = int(data.Forecast[i][0+k*4: 4+k*4])





            # statistical
            if model_name.strip() == "CPC CA":
                for k in range(9):
                    raw_CPC_CA[j+k, k] = int(data.Forecast[i][0+k*4: 4+k*4])

            if model_name.strip() == "CPC CCA":
                for k in range(9):
                    raw_CPC_CCA[j+k, k] = int(data.Forecast[i][0+k*4: 4+k*4])


            if model_name.strip() == "UCLA-TCD":
                for k in range(9):
                    raw_UCLA_TCD[j+k, k] = int(data.Forecast[i][0+k*4: 4+k*4])


            if model_name.strip() == "UBC NNET":
                for k in range(9):
                    raw_UBC_NNET[j+k, k] = int(data.Forecast[i][0+k*4: 4+k*4])

            # CPC MRKOV has forecasts for all times
            # for issued forecasts from 2002-02 till 2019-07
            # therefore it is last and with the j+=1 in this condition
            if model_name.strip() == "CPC MRKOV":
                for k in range(9):
                    raw_CPC_MRKOV[j+k, k] = int(data.Forecast[i][0+k*4: 4+k*4])

                j+=1


    except ValueError:
        pass



target_season = pd.date_range(start='2002-04-01', end='2019-10-01', freq='MS')
lead_time = np.arange(0, 9)

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

ds=ds.where(ds!=-999)
ds = ds/100
ds.to_netcdf(join(postdir, f'other_forecasts.nc'))