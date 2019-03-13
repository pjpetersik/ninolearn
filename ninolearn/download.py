from __future__ import print_function

from ftplib import FTP
from requests import get  
from os.path import isfile, join, exists
from os import remove, mkdir
import gzip
import shutil

from ninolearn.private import datadir

if not exists(datadir):
    print("make a data directory at %s" %datadir)
    mkdir(datadir)

def downloadFileFTP(info_dict, outdir='', username='anonymous', password='anonymous_pass'):
    """
    Download a file from a FTP server
    """
    filename = info_dict['filename']
    host = info_dict['host']
    location = info_dict['location']
    if not exists(join(datadir,outdir)):
        print("make a data directory at %s" %join(datadir,outdir))
        mkdir(join(datadir,outdir))
    
    if not isfile(join(datadir,outdir,filename)):
        print("Download %s" % filename)
        ftp = FTP(host)
        ftp.login(username,password)
        ftp.cwd(location)
        
        localfile = open(join(datadir,outdir,filename), 'wb')
        ftp.retrbinary('RETR ' + filename, localfile.write,1024)
        localfile.close()
        ftp.quit() 
    
    else:
        print("%s already downloaded" % filename)

def downloadFileHTTP(info_dict, outdir=''):
    """
    download a file via HTTP
    """
    # open in binary mode
    filename = info_dict['filename']
    url = info_dict['url']
   
    if not isfile(join(datadir,outdir,filename.replace(".gz",""))):
        print("Download %s" % filename)
        with open(join(datadir,outdir,filename), "wb") as file:
            # get request
            response = get(url)
            
            # write to file
            file.write(response.content)
    else:
       print("%s already downloaded" % filename) 
            
def unzip_gz(info_dict):
    """
    unzip .gz format files
    """
    filename_old = info_dict['filename']
    filename_new = filename_old.replace(".gz","")
    
    # unpack file
    if not isfile(join(datadir,filename_new)):
        print("Unzip %s" % filename_old)
        with gzip.open(join(datadir,filename_old), 'rb') as f_in:
            with open(join(datadir,filename_new), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
         print("%s already unzipped" % filename_old)
    
    # remove .gz file
    if isfile(join(datadir,filename_new)) and isfile(join(datadir,filename_old)):
        remove(join(datadir,filename_old))
        print("Remove %s " % filename_old)




if __name__ == "__main__":
    
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
    