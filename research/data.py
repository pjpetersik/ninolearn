from ninolearn.download import downloadFileFTP, downloadFileHTTP, unzip_gz
from ninolearn.private import CMEMS_password, CMEMS_username
from ninolearn.utils import print_header

# =============================================================================
# =============================================================================
# # Download
# =============================================================================
# =============================================================================
print_header("Download Data")

# =============================================================================
# ERSSTv5
# =============================================================================
ERSSTv5_dict = {
        'filename' : 'sst.mnmean.nc',
        'host' : 'ftp.cdc.noaa.gov',
        'location' : '/Datasets/noaa.ersst.v5/'
        }

downloadFileFTP(ERSSTv5_dict)

# =============================================================================
# NINO3.4 Index
# =============================================================================
NINO34_dict = { 
        'url' :'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt',
        'filename' : 'nino34.txt'
        }

downloadFileHTTP(NINO34_dict)

# =============================================================================
# HadISST1
# =============================================================================

HadISST1_dict = {
        'filename' : 'HadISST_sst.nc.gz',
        'url':'https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz'
        }

downloadFileHTTP(HadISST1_dict)
unzip_gz(HadISST1_dict)

# =============================================================================
# ORAP5.0 SSH
# =============================================================================
ORAP50_dict = {
    'filename' : 'sossheig_ORAP5.0_1m_197901_grid_T_02.nc',
    'host' : 'nrtcmems.mercator-ocean.fr',
    'location' : '/Core/GLOBAL_REANALYSIS_PHYS_001_017/global-reanalysis-phys-001-017-ran-uk-orap5.0-ssh/'
    }

for year_int in range(1979,2014):
    year_str = str(year_int)
    for month_int in range(1,13):
        if month_int<10:
            month_str = "0"+str(month_int)
        else:
            month_str = month_int
        
        ORAP50_dict['filename'] = 'sossheig_ORAP5.0_1m_%s%s_grid_T_02.nc'%(year_str,month_str)
        downloadFileFTP(ORAP50_dict,outdir='ssh', username=CMEMS_username,password=CMEMS_password)
        
        
# =============================================================================
# WWV 
# =============================================================================
WWV_dict = { 
        'filename': 'wwv.dat',
        'url' : 'https://www.pmel.noaa.gov/tao/wwv/data/wwv.dat'
        }

downloadFileHTTP(WWV_dict)        

# =============================================================================
# Wind
# =============================================================================
uwind_dict = {
        'filename' : 'uwnd.mon.mean.nc',
        'host' : 'ftp.cdc.noaa.gov',
        'location' : '/Datasets/ncep.reanalysis.derived/surface/'
        }

vwind_dict = {
        'filename' : 'vwnd.mon.mean.nc',
        'host' : 'ftp.cdc.noaa.gov',
        'location' : '/Datasets/ncep.reanalysis.derived/surface/'
        }

downloadFileFTP(uwind_dict)
downloadFileFTP(vwind_dict)

# =============================================================================
# Surface Air Temperature (SAT)
# =============================================================================

SAT_dict = {
        'filename' : 'air.sig995.2019.nc',
        'host' : 'ftp.cdc.noaa.gov',
        'location' : '/Datasets/ncep.reanalysis.dailyavgs/surface/'
        }

for year_int in range(1948,2019):
    year_str = str(year_int)
    SAT_dict['filename'] = 'air.sig995.%s.nc'%year_str
    downloadFileFTP(SAT_dict,outdir='sat')

# =============================================================================
# =============================================================================
# # Postprocess
# =============================================================================
# =============================================================================
print_header("Postprocess Data")
from ninolearn.postprocess.time_axis import add_DatetimeIndex_nino34, add_DatetimeIndex_wwv

add_DatetimeIndex_nino34()
add_DatetimeIndex_wwv()

from ninolearn.IO.read_raw import sst_ERSSTv5
from ninolearn.postprocess.statisitcs import postprocess

# postprocess sst
sst = sst_ERSSTv5()
postprocess(sst)
