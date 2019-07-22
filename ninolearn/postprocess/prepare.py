import pandas as pd
from os.path import join, exists
from os import mkdir
import numpy as np

from ninolearn.IO import read_raw
from ninolearn.pathes import postdir

from ninolearn.IO.read_post import data_reader

if not exists(postdir):
    print("make a data directory at %s" % postdir)
    mkdir(postdir)


def season_to_month(season):
    """
    translates a 3-month season string to the corresponding integer of the
    last month of the season (to ensure not to include any future information
    when predictions are made later with this data)

    :type season: string
    :param season: Season represented by three letters such as 'DJF'
    """
    switcher = {'DJF': 2,
                'JFM': 3,
                'FMA': 4,
                'MAM': 5,
                'AMJ': 6,
                'MJJ': 7,
                'JJA': 8,
                'JAS': 9,
                'ASO': 10,
                'SON': 11,
                'OND': 12,
                'NDJ': 1,
                }

    return switcher[season]

def season_shift_year(season):
    """
    when the function .season_to_month() is applied the year related to
    NDJ needs to be shifted by 1.

    :type season: string
    :param season: Season represented by three letters such as 'DJF'
    """
    switcher = {'DJF': 0,
                'JFM': 0,
                'FMA': 0,
                'MAM': 0,
                'AMJ': 0,
                'MJJ': 0,
                'JJA': 0,
                'JAS': 0,
                'ASO': 0,
                'SON': 0,
                'OND': 0,
                'NDJ': 1,
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

    df = ({'year': data.YR.values + data.SEAS.apply(season_shift_year).values,
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

def prep_wwv(cardinal_direction=""):
    """
    Add a time axis corresponding to the first day of the central month of a
    3-month season. For example: DJF 2019 becomes 2019-01-01. Further, rename
    some axis.
    """
    print(f"Prepare WWV {cardinal_direction} timeseries.")
    data = read_raw.wwv_anom(cardinal_direction=cardinal_direction)

    df = ({'year': data.date.astype(str).str[:4],
           'month': data.date.astype(str).str[4:],
           'day': data.date/data.date})
    dti = pd.to_datetime(df)

    data.index = dti
    data.index.name = 'time'
    data = data.rename(index=str, columns={'Anomaly': 'anom'})
    data.to_csv(join(postdir, f'wwv{cardinal_direction}.csv'))

def prep_K_index():
    """
    function that edits the Kirimati index from Bunge and Clarke (2014)
    """
    data = read_raw.K_index()
    data.index.name = 'time'
    data.name = 'anom'
    data.to_csv(join(postdir, f'kindex.csv'), header=True)

def prep_wwv_proxy():
    """
    Make a wwv proxy index that uses the K-index from Bunge and Clarke (2014)
    for the time period between 1955 and 1979
    """

    reader_wwv = data_reader(startdate='1980-01', enddate='2018-12')
    wwv = reader_wwv.read_csv('wwv')

    reader_kindex = data_reader(startdate='1955-01', enddate='1979-12')
    kindex = reader_kindex.read_csv('kindex') * 10e12

    wwv_proxy = kindex.append(wwv)
    wwv_proxy.to_csv(join(postdir, f'wwv_proxy.csv'), header=True)


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

def calc_warm_pool_edge():
    """
    calculate the warm pool edge
    """
    reader = data_reader(startdate='1948-01', enddate='2018-12',lon_min=120, lon_max=290)
    sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='')

    sst_eq = sst.loc[dict(lat=0)]
    warm_pool_edge = np.zeros(sst_eq.shape[0])
    indeces = np.zeros(sst_eq.shape[0])

    # TODO  not very efficent
    for i in range(sst_eq.shape[0]):
        index = np.argwhere(sst_eq[i].values>28.).max()
        indeces[i] = index

        slope = sst_eq[i, index] - sst_eq[i, index-1]

        intercept28C = (sst_eq[i, index] - 28.) * slope + index

        warm_pool_edge[i] = intercept28C * 2.5 * 111.321

    df = pd.DataFrame(data=warm_pool_edge,index=sst.time.values, columns=['total'])
    df.index.name = 'time'

    df.to_csv(join(postdir, 'wp_edge.csv'))
    return warm_pool_edge, indeces
