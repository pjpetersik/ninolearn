from os.path import join
import pandas as pd
import xarray as xr
import gc

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
        
    def __del__(self):
        gc.collect()
        
    def nino34_anom(self):
        """
        get the Nino3.4 Index anomaly
        """        
        data = pd.read_csv(join(postdir,"nino34.csv"),index_col=0, parse_dates=True)
        self._check_dates(data, "Nino3.4")
        
        return data.ANOM.loc[self.startdate:self.enddate]
    
    def wwv_anom(self):
        """
        get the warm water volume anomaly
        """
        data = pd.read_csv(join(postdir,"wwv.csv"),index_col=0, parse_dates=True)
        self._check_dates(data, "WWV")
        
        return data.Anomaly.loc[self.startdate:self.enddate]
    
    def sst_ERSSTv5(self):
        """
        get the sea surface temperature from the ERSST-v5 data set
        """
        data = xr.open_dataset(join(rawdir,"sst.mnmean.nc"))
        self._check_dates(data.sst, "SST (ERSSTv5)")
        
        return data.sst.loc[self.startdate:self.enddate,
                            self.lat_max:self.lat_min,
                            self.lon_min:self.lon_max]
    
    def sst_HadISST(self):
        """
        get the sea surface temperature from the ERSST-v5 data set
        
        NOT FULLY POSTPROCESSED YET
        """
        print("HadISST data is NOT FULLY POSTPROCESSED YET!")
        data = xr.open_dataset(join(rawdir,"HadISST_sst.nc"))
        self._check_dates(data.sst, "SST (HadISST)")
        return data
    
    def uwind(self):
        """
        get u-wind from NCEP/NCAR reanalysis
        """
        data = xr.open_dataset(join(rawdir,"uwnd.mon.mean.nc"))
        self._check_dates(data.uwnd, "uwind (NCEP/NCAR)")
        
        return data.uwnd.loc[self.startdate:self.enddate,
                            self.lat_max:self.lat_min,
                            self.lon_min:self.lon_max]
    
    def vwind(self):
        """
        get v-wind from NCEP/NCAR reanalysis
        """
        data = xr.open_dataset(join(rawdir,"vwnd.mon.mean.nc"))
        self._check_dates(data.vwnd, "vwind (NCEP/NCAR)")
        
        return data.vwnd.loc[self.startdate:self.enddate,
                            self.lat_max:self.lat_min,
                            self.lon_min:self.lon_max]
    
    
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
    sst_ERSST = reader.sst_ERSSTv5()
    #sst_HadISST = reader.sst_HadISST()
    uwind = reader.uwind()
    vwind = reader.vwind()