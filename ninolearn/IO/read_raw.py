from os.path import join
import pandas as pd
import xarray as xr

from ninolearn.pathes import rawdir

def nino34_anom():
    """
    get the Nino3.4 Index anomaly
    """        
    data = pd.read_csv(join(rawdir,"nino34.txt"), delim_whitespace=True)
    return data

def wwv_anom():
    """
    get the warm water volume anomaly
    """
    data = pd.read_csv(join(rawdir,"wwv.dat"), delim_whitespace=True, header=4)
    return data

def sst_ERSSTv5():
    """
    get the sea surface temperature from the ERSST-v5 data set
    """
    data = xr.open_dataset(join(rawdir,'sst.mnmean.nc'))            
    return data.sst
        

def sst_HadISST():
    """
    get the sea surface temperature from the ERSST-v5 data set and directly 
    manipulate the time axis in such a way that the monthly mean values are assigned
    to the beginning of a month as this is the default for the other data sets
    """
    data = xr.open_dataset(join(rawdir,"HadISST_sst.nc"))
    maxtime = pd.to_datetime(data.time.values.max()).date()
    data['time'] = pd.date_range(start='1870-01-01', end=maxtime, freq='MS')
    return data.sst

def uwind():
    """
    get u-wind from NCEP/NCAR reanalysis
    """
    data = xr.open_dataset(join(rawdir,"uwnd.mon.mean.nc"))
    return data

def vwind():
    """
    get v-wind from NCEP/NCAR reanalysis
    """
    data = xr.open_dataset(join(rawdir,"vwnd.mon.mean.nc"))
    return data

def sat():
    """
    Get the surface air temperature from NCEP/NCAR Reanalysis
    """
    data = xr.open_mfdataset(join(rawdir,'sat','*.nc'))
    return data.air

def ssh():
    data = xr.open_mfdataset(join(rawdir,'ssh','*.nc'))
    return data