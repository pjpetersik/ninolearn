"""
This module contains methods to download files from ftp-servers or via http.
The method download()
"""

from ftplib import FTP
from requests import get
from os.path import isfile, join, exists
from os import remove, mkdir
import gzip
import shutil

from ninolearn.pathes import rawdir

if not exists(rawdir):
    print("make a data directory at %s" % rawdir)
    mkdir(rawdir)

def download(info_dict, **kwargs):
    """
    Download data specified in the dictionary *info_dict*. The value of the
    keyword *'downloadType' is important to tell the method if the download
    is done from an ftp-server (value *'ftp'*) or via http (value *'http'*).

    The information dictionary  *'info_dict'* contains the essential
    informations over the source of the data. See
    :class:`ninolearn.download.sources` for a detailed description how the
    information dictionary should be orgainized.

    :type info_dict: dict
    :param info_dict: The information dictionary for the source of the data.\
    See :class:`ninolearn.download.sources` for more details on the organization\
    of the dictionary.

    :param kwargs: Keyword arguments that are passed to :method:'ninolearn.download.downloadFTP' or\
    :method:'ninolearn.download.downloadHTTP'.
    """
    if info_dict['downloadType']=='ftp':
        downloadFTP(info_dict, **kwargs)

    elif info_dict['downloadType']=='http':
        downloadHTTP(info_dict, **kwargs)


def downloadFTP(info_dict, outdir='',
                    username='anonymous', password='anonymous_pass'):
    """
    Download a file from a FTP server. Note, just some ftp-servers
    require an login account. Make sure you do **NOT commit** code in which your
    login details are visible to a public repository. Hence, put the information
    in a module that that you do not commit (put its name into the .gitignore)
    file.

    :param info_dict: The information dictionary for the source of the data.\
    See :class:'ninolearn.download.sources' for more details on the organization\
    of the dictionary.

    :type outdir: str
    :param outdir: The output directory for your variable. If outdir='', then\
    no separate directory is made and all data is put into the raw data\
    directory (see ninolearn.pathes).

    :type username: str
    :param username: Username for ftp-server login (not always required).

    :type password: str
    :param password: Password for ftp-server login (not always required).
    """
    filename = info_dict['filename']
    host = info_dict['host']
    location = info_dict['location']

    # make the output directory for the file
    if not exists(join(rawdir, outdir)):
        print("make a data directory at %s" % join(rawdir, outdir))
        mkdir(join(rawdir, outdir))


    if not isfile(join(rawdir, outdir, filename)):
        print("Download %s" % filename)
        ftp = FTP(host)
        ftp.login(username, password)
        ftp.cwd(location)

        localfile = open(join(rawdir, outdir, filename), 'wb')
        ftp.retrbinary('RETR ' + filename, localfile.write, 204800)
        localfile.close()
        ftp.quit()

    else:
        print("%s already downloaded" % filename)


def downloadHTTP(info_dict, outdir=''):
    """
    Download a file via a HTTP.

    :type info_dict: dict
    :param info_dict: The information dictionary for the source of the data.\
    See :class:`ninolearn.download.sources` for more details on the organization\
    of the dictionary.

    :type outdir: str
    :param outdir: The output directory for your variable. If *outdir=''*, then\
    no separate directory is made and all data is put into the raw data \
    directory (see ninolearn.pathes).
    """
    # open in binary mode
    filename = info_dict['filename']
    url = info_dict['url']

    if not isfile(join(rawdir, outdir, filename.replace(".gz", ""))):
        print("Download %s" % filename)
        with open(join(rawdir, outdir, filename), "wb") as file:
            # get request
            response = get(url)

            # write to file
            file.write(response.content)
    else:
        print("%s already downloaded" % filename)


def unzip_gz(info_dict):
    """
    Unzip .gz format file. Some downloaded files come in in zipped format.
    With this method you can unzip them directly after the download and remove
    the .gz file immediately.

    :type info_dict: dict
    :param info_dict: The information dictionary for the source of the data.\
    See :class:`ninolearn.download.sources` for more details on the organization\
    of the dictionary.
    """
    filename_old = info_dict['filename']
    filename_new = filename_old.replace(".gz", "")

    # unpack file
    if not isfile(join(rawdir, filename_new)):
        print("Unzip %s" % filename_old)
        with gzip.open(join(rawdir, filename_old), 'rb') as f_in:
            with open(join(rawdir, filename_new), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print("%s already unzipped" % filename_old)

    # remove .gz file
    if isfile(join(rawdir, filename_new)) \
       and isfile(join(rawdir, filename_old)):
        remove(join(rawdir, filename_old))
        print("Remove %s " % filename_old)



class sources(object):
    """
    Collection of relevant data sources. If you add a new data source yourself
    follow the dictionary template:

    For FTP download, you need to specify the host (keyword 'host') the directory
    on the ftp-server (keyword 'location') as well as the filename (keyword 'filename')
    that is going to be the name of the file AS WELL AS for the downloaded file.

    **NAME_FTP** = {\n
        'downloadType':'ftp',\n
        'filename': 'filename_on_the_server_AND_local',\n
        'host': 'ftp.hostname.com',\n
        'location': '/directory/on/the/server'\n
        }


    For HTTP download, you need to specific the FULL path to the file (keyword 'url')
    as well as the name that is used for the downloaded file (keyword 'filename')

    **NAME_HTTP** = {\n
           'downloadType':'http',\n
           'url': 'https://www.full_url_to_the_file.com/data.txt',\n
           'filename': 'local_file_name.txt'\n
            }

    **Source dictionaries:**\n

    :Source ONI: The Oceanic Nino Index.

    :Source NINOindeces: The monthly Nino1+2, Nino3, Nino4 and Nino3.4 indeces.

    :Source WWV,WWV_West: The Warm water volume (WWV) and the WWV in the\
    western basin.

    :Source IOD: The  Dipole Mode Index (DMI) of the Indian Ocean Dipole  (IOD).

    :Source SST_ERSSTv5: The SST field from the ERSSTv5 field.

    :Source ORAS4: The ORAS4 data set. Define the argument for the keyword\
    *'filename'* yourself. There are various variables available form the \
    ORAS4 data set. Moreover, they are just available in multiple files (not \
    in a single big file).

    :Source GODAS: The GODAS data set. Define the argument for the keyword\
    *'filename'* yourself. Data is just available in multiple files (not \
    in a single big file).

    :Source UWIND_NCEP,VWIND_NCEP,SAT_monthly_NCEP: The *monthly* u-wind, vwind\
    and surface air temperature (SAT) from the NCEP reanalysis.

    :Source SAT_daily_NCEP: The *daily* surface air temperature (SAT) from \
    the NCEP reanalysis.

    :Source HCA: The heat content anomaly. (Data source: NOAA)

    :Source OLR_NOAA: The outgoing longwave radiation (OLR).\
    (Data source: NOAA)
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

    otherForecasts = {
            'downloadType':'http',
            'url': 'https://iri.columbia.edu/~forecast/ensofcst/Data/ensofcst_ALLto0719',
            'filename': 'other_forecasts.csv'

            }

