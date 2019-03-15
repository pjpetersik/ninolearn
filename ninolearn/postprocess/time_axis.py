import pandas as pd
from os.path import join, exists
from os import mkdir

from ninolearn.IO import read_raw
from ninolearn.pathes import rawdir, postdir

if not exists(postdir):
    print("make a data directory at %s" %postdir)
    mkdir(postdir)

def season_to_month(season):
    """
    translates a 3-month season string to the corresponding integer of the 
    central month
    
    :type season: string
    :param season: Season represented by three letters such as 'DJF'
    """
    switcher = {'DJF':1,
                'JFM':2,
                'FMA':3,
                'MAM':4,
                'AMJ':5,
                'MJJ':6,
                'JJA':7,
                'JAS':8,
                'ASO':9,
                'SON':10,
                'OND':11,
                'NDJ':12,                                
                }
    
    return switcher[season]

def add_DatetimeIndex_nino34():
    """
    Add a time axis corresponding to the first day of the central month of a 3-month season.
    For example: DJF 2019 becomes 2019-01-01
    """
    print("Postprocess Nino3.4 timeseries.")
    data = read_raw.nino34_anom() 
    
    df = ({'year': data.YR.values, 'month': data.SEAS.apply(season_to_month).values, 'day': data.YR.values/data.YR.values})
    dti = pd.to_datetime(df)
    
    data.index = dti
    
    data.to_csv(join(postdir,'nino34.csv'))

def add_DatetimeIndex_wwv():
    """
    Add a time axis corresponding to the first day of the central month of a 3-month season.
    For example: DJF 2019 becomes 2019-01-01
    """
    print("Postprocess WWV timeseries.")
    data = read_raw.wwv_anom() 
    
    df = ({'year': data.date.astype(str).str[:4], 'month': data.date.astype(str).str[4:], 'day': data.date/data.date})
    dti = pd.to_datetime(df)
    
    data.index = dti
    data.to_csv(join(postdir,'wwv.csv'))


if __name__ == "__main__":
    add_DatetimeIndex_nino34()
    add_DatetimeIndex_wwv()
    
    