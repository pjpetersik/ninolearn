import pandas as pd
from os.path import join, exists
from os import mkdir
import numpy as np

from ninolearn.IO import read_raw
from ninolearn.pathes import postdir

if not exists(postdir):
    print("make a data directory at %s" % postdir)
    mkdir(postdir)


def season_to_month(season):
    """
    translates a 3-month season string to the corresponding integer of the
    central month

    :type season: string
    :param season: Season represented by three letters such as 'DJF'
    """
    switcher = {'DJF': 1,
                'JFM': 2,
                'FMA': 3,
                'MAM': 4,
                'AMJ': 5,
                'MJJ': 6,
                'JJA': 7,
                'JAS': 8,
                'ASO': 9,
                'SON': 10,
                'OND': 11,
                'NDJ': 12,
                }

    return switcher[season]

def prep_nino_seasonal():
    """
    Add a time axis corresponding to the first day of the central month of a
    3-month season. For example: DJF 2019 becomes 2019-01-01. Further, rename
    some axis.
    """
    print("Prepare Nino3.4 timeseries.")
    index="3.4"
    period ="S"
    data = read_raw.nino_anom(index=index, period=period, detrend=False)

    df = ({'year': data.YR.values,
           'month': data.SEAS.apply(season_to_month).values,
           'day': data.YR.values/data.YR.values})
    dti = pd.to_datetime(df)

    data.index = dti
    data.index.name = 'time'
    data = data.rename(index=str, columns={'ANOM': 'anom'})
    data.to_csv(join(postdir, f'nino{index}{period}.csv'))

def prep_nino_month(index="3.4", detrend=False):
    """
    Add a time axis corresponding to the first day of the central month.
    """
    print("Prepare monthly Nino3.4 timeseries.")
    period ="M"

    rawdata = read_raw.nino_anom(index=index, period=period, detrend=detrend)
    rawdata = rawdata.rename(index=str, columns={'ANOM': 'anomNINO1+2',
                                                 'ANOM.1': 'anomNINO3',
                                                 'ANOM.2': 'anomNINO4',
                                                 'ANOM.3': 'anomNINO3.4'})

    dftime = ({'year': rawdata.YR.values,
           'month': rawdata.MON.values,
           'day': rawdata.YR.values/rawdata.YR.values})
    dti = pd.to_datetime(dftime)

    data = pd.DataFrame(data=rawdata[f"anomNINO{index}"])

    data.index = dti
    data.index.name = 'time'
    data = data.rename(index=str, columns={f'anomNINO{index}': 'anom'})

    filename = f"nino{index}{period}"

    if detrend:
        filename = ''.join(filename, "detrend")
    filename = ''.join((filename,'.csv'))

    data.to_csv(join(postdir, filename))

def prep_wwv():
    """
    Add a time axis corresponding to the first day of the central month of a
    3-month season. For example: DJF 2019 becomes 2019-01-01. Further, rename
    some axis.
    """
    print("Prepare WWV timeseries.")
    data = read_raw.wwv_anom()

    df = ({'year': data.date.astype(str).str[:4],
           'month': data.date.astype(str).str[4:],
           'day': data.date/data.date})
    dti = pd.to_datetime(df)

    data.index = dti
    data.index.name = 'time'
    data = data.rename(index=str, columns={'Anomaly': 'anom'})
    data.to_csv(join(postdir, 'wwv.csv'))

def prep_iod():
    """
    Prepare the IOD index dataframe
    """
    print("Prepare IOD timeseries.")

    data = read_raw.iod()
    data = data.T.unstack()
    data = data.replace(-999, np.nan)

    dti = pd.date_range(start='1870-01-01', end='2018-12-01', freq='MS')

    df = pd.DataFrame(data=data.values,index=dti, columns=['anom'])
    df.index.name = 'time'

    df.to_csv(join(postdir, 'iod.csv'))

if __name__ == "__main__":
#    prep_nino_seasonal()
#    prep_wwv()
#    prep_nino_month()
    a=prep_iod()