from os.path import join
import pandas as pd
import xarray as xr
from scipy.io import loadmat

from ninolearn.pathes import rawdir

"""
This module collects a bunch methods to read the raw data files.
"""


def nino34_anom():
    """
    Get the Nino3.4 Index anomaly.
    """
    data = pd.read_csv(join(rawdir, "nino34.txt"), delim_whitespace=True)
    return data

def nino_anom(index="3.4", period ="S", detrend=False):
    """
    read various Nino indeces from the raw directory
    """
    try:
        if period == "S":
            if index == "3.4" and not detrend:
                data = pd.read_csv(join(rawdir, "nino34.txt"),
                                   delim_whitespace=True)
            else:
                msg = "Only not detrended Nino3.4 index is available for seasonal records"
                raise Exception(msg)


        elif period == "M":
            if detrend and index == "3.4":
                data = pd.read_csv(join(rawdir, "nino34detrend.txt"),
                               delim_whitespace=True)
            elif not detrend:
                data = pd.read_csv(join(rawdir, "nino_1_4.txt"),
                                           delim_whitespace=True)
        return data

    except UnboundLocalError:
        raise Exception("The desired NINO index is not available.")


def wwv_anom(cardinal_direction=""):
    """
    get the warm water volume anomaly
    """
    if cardinal_direction != "":
        filename = f"wwv_{cardinal_direction}.dat"
    else:
        filename = "wwv.dat"

    data = pd.read_csv(join(rawdir, filename),
                       delim_whitespace=True, header=4)
    return data

def iod():
    """
    get IOD index data
    """
    data = pd.read_csv(join(rawdir, "iod.txt"),
                       delim_whitespace=True, header=None, skiprows=1, skipfooter=7,
                       index_col=0, engine='python')
    return data

def K_index():
    data = loadmat(join(rawdir, "Kindex.mat"))

    kindex = data['Kindex2_mon_anom'][:,0]
    time = pd.date_range(start='1955-01-01', end='2011-12-01', freq='MS')
    ds = pd.Series(data=kindex, index=time)
    return ds

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

def ustr():
    """
    get u-wind stress from ICOADS 1-degree Enhanced

    """
    data = xr.open_dataset(join(rawdir, "upstr.mean.nc"))
    data.upstr.attrs['dataset'] = 'ICOADS'
    return data.upstr

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


def olr():
    """
    get v-wind from NCEP/NCAR reanalysis
    """
    data = xr.open_dataset(join(rawdir, "olr.mon.mean.nc"))
    data.olr.attrs['dataset'] = 'NCAR'
    return data.olr


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

def godas(variable="sshg"):
    ds = xr.open_mfdataset(join(rawdir, f'{variable}_godas', '*.nc'),
                             concat_dim='time')

    if len(ds[variable].shape)==4:
        data = ds.loc[dict(level=5)].load()
    else:
        data = ds.load()

    data[variable].attrs['dataset'] = 'GODAS'
    return data[variable]

def oras4():
    ds = xr.open_mfdataset(join(rawdir, f'ssh_oras4', '*.nc'),
                             concat_dim='time')
    data = ds.load()
    data.zos.attrs['dataset'] = 'ORAS4'
    return data.zos

def sat_gfdl():
    data = xr.open_mfdataset(join(rawdir, 'sat_gfdl', '*.nc'),
                             concat_dim='time')

    data = data.load()
    data.tas.attrs['dataset'] = 'GFDL-CM3'

    # this change needs to be done to prevent OutOfBoundsError
    data['time'] = pd.date_range(start='1700-01-01', end='2199-12-01',freq='MS')
    return data.tas

def ssh_gfdl():
    data = xr.open_mfdataset(join(rawdir, 'ssh_gfdl', '*.nc'),
                             concat_dim='time')
    #data = data.load()
    data.zos.attrs['dataset'] = 'GFDL-CM3'

    # this change needs to be done to prevent OutOfBoundsError
    data['time'] = pd.date_range(start='1700-01-01', end='2199-12-01',freq='MS')
    return data.zos


def sst_gfdl():
    data = xr.open_mfdataset(join(rawdir, 'sst_gfdl', '*.nc'),
                             concat_dim='time')
    #data = data.load()
    data.tos.attrs['dataset'] = 'GFDL-CM3'

    # this change needs to be done to prevent OutOfBoundsError
    data['time'] = pd.date_range(start='1700-01-01', end='2199-12-01',freq='MS')
    return data.tos

def hca_mon():
    """
    heat content anomaly, seasonal variable to the first day of the middle season
    and upsample the data
    """
    data = xr.open_dataset(join(rawdir, "hca.nc"), decode_times=False)
    data['time'] = pd.date_range(start='1955-02-01', end='2019-02-01', freq='3MS')
    data.h18_hc.attrs['dataset'] = 'NODC'

    data_raw = data.h18_hc[:,0,:,:]
    data_upsampled = data_raw.resample(time='MS').interpolate('linear')
    data_upsampled.name = 'hca'
    return data_upsampled
