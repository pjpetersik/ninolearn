from os.path import join
import pandas as pd
import xarray as xr

from ninolearn.private import datadir

class data_reader(object):
    def __init__(self, startyear = 1980, endyear = 2010, lon_min = 120, lon_max = 260, lat_min = -30, lat_max = 30):
        """
        Data reader for different kind of El Nino related data.
        
        :param startyear: year from which on data should be loaded (from January on)
        :param endyear: year to which data should be loaded (until December)
        :lon_min: eastern boundary of data set in degrees east
        :lon_max: western boundary of data set in degrees east
        :lat_min: southern boundary of data set in degrees north
        :lat_max: northern boundary of data set in degrees north
        """
        self.startyear = startyear
        self.endyear = endyear
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        
    def nino34_anom(self):
        """
        get the Nino3.4 Index anomaly
        """
        data = pd.read_csv(join(datadir,"nino34.txt"), delim_whitespace=True)
        return data.ANOM[(data.YR>=self.startyear) & (data.YR>=self.startyear)]
    
    def wwv_anom(self):
        """
        get the warm water volume anomaly
        """
        data = pd.read_csv(join(datadir,"wwv.dat"), delim_whitespace=True, header=4)
        return data.Anomaly
    
    def sst_ERSSTv5(self):
        """
        get the sea surface temperature from the ERSST-v5 data set
        """
        data = xr.open_dataset(join(datadir,"sst.mnmean.nc"))
        return data
    
    def sst_HadISST(self):
        """
        get the sea surface temperature from the ERSST-v5 data set
        """
        data = xr.open_dataset(join(datadir,"HadISST_sst.nc"))
        return data
    
    def uwind(self):
        """
        get u-wind from NCEP/NCAR reanalysis
        """
        data = xr.open_dataset(join(datadir,"uwnd.mon.mean.nc"))
        return data
    
    def vwind(self):
        """
        get u-wind from NCEP/NCAR reanalysis
        """
        data = xr.open_dataset(join(datadir,"vwnd.mon.mean.nc"))
        return data

if __name__ == "__main__":
    reader = data_reader()
    data = reader.nino34_anom()
    data2 = reader.wwv_anom()
    data3 = reader.sst_ERSSTv5()
    data4 = reader.uwind()
    data5 = reader.vwind()