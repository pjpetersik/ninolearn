from ninolearn.download import downloadFileFTP, downloadFileHTTP, unzip_gz
from ninolearn.private import CMEMS_password, CMEMS_username

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
