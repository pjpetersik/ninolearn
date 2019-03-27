from os.path import join
import pandas as pd
import xarray as xr

from ninolearn.pathes import rawdir


def nino34_anom():
    """
    get the Nino3.4 Index anomaly
    """
    data = pd.read_csv(join(rawdir, "nino34.txt"), delim_whitespace=True)
    return data


def wwv_anom():
    """
    get the warm water volume anomaly
    """
    data = pd.read_csv(join(rawdir, "wwv.dat"),
                       delim_whitespace=True, header=4)
    return data


def sst_ERSSTv5():
    """
    get the sea surface temperature from the ERSST-v5 data set
    """
    data = xr.open_dataset(join(rawdir, 'sst.mnmean.nc'))
    data.sst.attrs['dataset'] = 'ERSSTv5'
    return data.sst


def sst_HadISST():
    """
    get the sea surface temperature from the ERSST-v5 data set and directly
    manipulate the time axis in such a way that the monthly mean values are
    assigned to the beginning of a month as this is the default for the other
    data sets
    """
    data = xr.open_dataset(join(rawdir, "HadISST_sst.nc"))
    maxtime = pd.to_datetime(data.time.values.max()).date()
    data['time'] = pd.date_range(start='1870-01-01', end=maxtime, freq='MS')
    data.sst.attrs['dataset'] = 'HadISST'
    return data.sst


def uwind():
    """
    get u-wind from NCEP/NCAR reanalysis
    """
    data = xr.open_dataset(join(rawdir, "uwnd.mon.mean.nc"))
    data.uwnd.attrs['dataset'] = 'NCEP'
    return data.uwnd


def vwind():
    """
    get v-wind from NCEP/NCAR reanalysis
    """
    data = xr.open_dataset(join(rawdir, "vwnd.mon.mean.nc"))
    data.vwnd.attrs['dataset'] = 'NCEP'
    return data.vwnd


def sat(mean='monthly'):
    """
    Get the surface air temperature from NCEP/NCAR Reanalysis

    :param mean: Choose between daily and monthly mean fields
    """
    if mean == 'monthly':
        data = xr.open_dataset(join(rawdir, "air.mon.mean.nc"))
        data.air.attrs['dataset'] = 'NCEP'
        return data.air

    elif mean == 'daily':
        data = xr.open_mfdataset(join(rawdir, 'sat', '*.nc'))
        data_return = data.air

        data_return.attrs['dataset'] = 'NCEP'
        data_return.name = 'air_daily'
        return data_return


def ssh():
    """
    Get sea surface height. And change some attirbutes and coordinate names
    """
    data = xr.open_mfdataset(join(rawdir, 'ssh', '*.nc'),
                             concat_dim='time_counter')
    data_return = data.sossheig.rename({'time_counter': 'time'})
    maxtime = pd.to_datetime(data_return.time.values.max()).date()
    data_return['time'] = pd.date_range(start='1979-01-01',
                                        end=maxtime,
                                        freq='MS')
    data_return.attrs['dataset'] = 'ORAP5'
    data_return.name = 'ssh'
    return data_return
