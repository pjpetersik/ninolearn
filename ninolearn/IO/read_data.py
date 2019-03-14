from os.path import join
import pandas as pd
import xarray as xr

from ninolearn.pathes import rawdir, postdir


class data_reader(object):
    def __init__(self, startdate = '1980-01', enddate = '2018-12', lon_min = 120, lon_max = 260, lat_min = -30, lat_max = 30):
        """
        Data reader for different kind of El Nino related data.
        
        :param startdate:year and month from which on data should be loaded 
        :param enddate: year and month to which data should be loaded 
        :lon_min: eastern boundary of data set in degrees east
        :lon_max: western boundary of data set in degrees east
        :lat_min: southern boundary of data set in degrees north
        :lat_max: northern boundary of data set in degrees north
        """
        self.startdate = startdate + "-01"
        self.enddate = enddate + "-01"
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        
    def nino34_anom(self):
        """
        get the Nino3.4 Index anomaly
        """        
        data = pd.read_csv(join(postdir,"nino34.csv"),index_col=0, parse_dates=True)
        self._check_dates(data, "Nino3.4")
        
        return data.ANOM
    
    def wwv_anom(self):
        """
        get the warm water volume anomaly
        """
        data = pd.read_csv(join(postdir,"wwv.csv"),index_col=0, parse_dates=True)
        self._check_dates(data, "WWV")
        
        return data.Anomaly
    
    def sst_ERSSTv5(self):
        """
        get the sea surface temperature from the ERSST-v5 data set
        """
        data = xr.open_dataset(join(rawdir,"sst.mnmean.nc"))
        return data
    
    def sst_HadISST(self):
        """
        get the sea surface temperature from the ERSST-v5 data set
        """
        data = xr.open_dataset(join(rawdir,"HadISST_sst.nc"))
        return data
    
    def uwind(self):
        """
        get u-wind from NCEP/NCAR reanalysis
        """
        data = xr.open_dataset(join(rawdir,"uwnd.mon.mean.nc"))
        return data
    
    def vwind(self):
        """
        get u-wind from NCEP/NCAR reanalysis
        """
        data = xr.open_dataset(join(rawdir,"vwnd.mon.mean.nc"))
        return data
    
    
    def _check_dates(self,data, name):
        """
        Checks if provided start and end date are in the bounds of the data that 
        should be read.
        """
        try:
            data.loc[self.startdate]
        except:
            raise IndexError("The startdate is out of bounds for %s data!"%name) 
        
        try:
            data.loc[self.enddate]
        except:
            raise IndexError("The enddate is out of bounds for %s data!"%name)

if __name__ == "__main__":
    reader = data_reader()
    nino34 = reader.nino34_anom()
    wwv = reader.wwv_anom()
    data3 = reader.sst_ERSSTv5()
    data4 = reader.uwind()
    data5 = reader.vwind()