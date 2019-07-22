"""
Collection of relevant data sources. If you add a new data source yourself
follow the dictionary template:

For FTP download, you need to specify the host (keyword 'host') the directory
on the ftp-server (keyword 'location') as well as the filename (keyword 'filename')
that is going to be the name of the file AS WELL AS for the downloaded file.

NAME = {
    'downloadType':'ftp',
    'filename': 'filename_on_the_server_AND_local',
    'host': 'ftp.hostname.com',
    'location': '/directory/on/the/server'
    }


For HTTP download, you need to specific the FULL path to the file (keyword 'url')
as well as the name that is used for the downloaded file (keyword 'filename')

ONI = {
       'downloadType':'http',
       'url': 'https://www.full_url_to_the_file.com/data.txt,
       'filename': 'local_file_name.txt'
        }

"""

SST_ERSSTv5 = {
        'downloadType':'ftp',
        'filename': 'sst.mnmean.nc',
        'host': 'ftp.cdc.noaa.gov',
        'location': '/Datasets/noaa.ersst.v5/'
        }

ONI = {
       'downloadType':'http',
       'url': 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt',
       'filename': 'oni.txt'
        }

NINO34detrend = {
        'downloadType':'http',
        'url': 'https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt',
        'filename': 'nino34detrend.txt'
        }

NINOindeces = {
        'downloadType':'http',
        'url': 'https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.81-10.ascii',
        'filename': 'nino_1_4.txt'
        }


ORAS4 = {
        'downloadType':'ftp',
        'filename': 'zos_oras4_1m_1958_grid_1x1.nc',
        'host': 'ftp-icdc.cen.uni-hamburg.de',
        'location': '/EASYInit/ORA-S4/monthly_1x1/'
              }

GODAS = {
        'downloadType':'ftp',
        'filename': 'sshg.1980.nc',
        'host': 'ftp.cdc.noaa.gov',
        'location': '/Datasets/godas/'
              }

WWV = {
        'downloadType':'http',
        'filename': 'wwv.dat',
        'url': 'https://www.pmel.noaa.gov/tao/wwv/data/wwv.dat'
        }

WWV_West = {
        'downloadType':'http',
        'filename': 'wwv_west.dat',
        'url': 'https://www.pmel.noaa.gov/tao/wwv/data/wwv_west.dat'
        }

UWIND_NCEP = {
        'downloadType':'ftp',
        'filename': 'uwnd.mon.mean.nc',
        'host': 'ftp.cdc.noaa.gov',
        'location': '/Datasets/ncep.reanalysis.derived/surface/'
        }

VWIND_NCEP = {
        'downloadType':'ftp',
        'filename': 'vwnd.mon.mean.nc',
        'host': 'ftp.cdc.noaa.gov',
        'location': '/Datasets/ncep.reanalysis.derived/surface/'
        }


SAT_daily_NCEP = {
        'downloadType':'ftp',
        'filename': 'air.sig995.2019.nc',
        'host': 'ftp.cdc.noaa.gov',
        'location': '/Datasets/ncep.reanalysis.dailyavgs/surface/'
        }

SAT_monthly_NCEP = {
        'downloadType':'ftp',
        'filename': 'air.mon.mean.nc',
        'host': 'ftp.cdc.noaa.gov',
        'location': '/Datasets/ncep.reanalysis.derived/surface/'
        }

IOD = {
        'downloadType':'http',
        'url': 'https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Data/dmi.long.data',
        'filename': 'iod.txt'
        }


#https://www.ncdc.noaa.gov/cdr/oceanic/ocean-heat-content
HCA = {
        'downloadType':'http',
        'filename': 'hca.nc',
        'url': 'http://data.nodc.noaa.gov/woa/DATA_ANALYSIS/3M_HEAT_CONTENT/NETCDF/heat_content/heat_content_anomaly_0-700_seasonal.nc'
        }

OLR_NOAA = {
        'downloadType':'ftp',
        'filename': 'olr.mon.mean.nc',
        'host': 'ftp.cdc.noaa.gov',
        'location': '/Datasets/interp_OLR/'
        }

