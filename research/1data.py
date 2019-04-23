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
        'filename': 'sst.mnmean.nc',
        'host': 'ftp.cdc.noaa.gov',
        'location': '/Datasets/noaa.ersst.v5/'
        }

downloadFileFTP(ERSSTv5_dict)

# =============================================================================
# NINO3.4 Index
# =============================================================================
NINO34_dict = {
        'url': 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt',
        'filename': 'nino34.txt'
        }

NINO34detrend_dict = {
        'url': 'https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt',
        'filename': 'nino34detrend.txt'
        }

NINOindeces_dict = {
        'url': 'https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.81-10.ascii',
        'filename': 'nino_1_4.txt'
        }


downloadFileHTTP(NINO34_dict)
downloadFileHTTP(NINO34detrend_dict)
downloadFileHTTP(NINOindeces_dict)

# =============================================================================
# HadISST1
# =============================================================================

HadISST1_dict = {
    'filename': 'HadISST_sst.nc.gz',
    'url': 'https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz'
        }

downloadFileHTTP(HadISST1_dict)
unzip_gz(HadISST1_dict)

# =============================================================================
# ORAP5.0 SSH
# =============================================================================
ORAP50_dict = {
    'filename': 'sossheig_ORAP5.0_1m_197901_grid_T_02.nc',
    'host': 'nrtcmems.mercator-ocean.fr',
    'location': '/Core/GLOBAL_REANALYSIS_PHYS_001_017/\
    global-reanalysis-phys-001-017-ran-uk-orap5.0-ssh/'
    }

for year_int in range(1979, 2014):
    year_str = str(year_int)
    for month_int in range(1, 13):
        if month_int < 10:
            month_str = "0"+str(month_int)
        else:
            month_str = month_int

        ORAP50_dict['filename'] = 'sossheig_ORAP5.0_1m_%s%s_grid_T_02.nc'\
            % (year_str, month_str)
        downloadFileFTP(ORAP50_dict, outdir='ssh',
                        username=CMEMS_username, password=CMEMS_password)

GODAS_dict = {'filename': 'sshg.1980.nc',
              'host': 'ftp.cdc.noaa.gov',
              'location': '/Datasets/godas/'
              }

for i in range(1980,2019):
    GODAS_dict['filename'] = f'sshg.{i}.nc'
    downloadFileFTP(GODAS_dict, outdir = 'ssh_godas')
# =============================================================================
# WWV
# =============================================================================
WWV_dict = {
        'filename': 'wwv.dat',
        'url': 'https://www.pmel.noaa.gov/tao/wwv/data/wwv.dat'
        }

downloadFileHTTP(WWV_dict)

# =============================================================================
# Wind
# =============================================================================
uwind_dict = {
        'filename': 'uwnd.mon.mean.nc',
        'host': 'ftp.cdc.noaa.gov',
        'location': '/Datasets/ncep.reanalysis.derived/surface/'
        }

vwind_dict = {
        'filename': 'vwnd.mon.mean.nc',
        'host': 'ftp.cdc.noaa.gov',
        'location': '/Datasets/ncep.reanalysis.derived/surface/'
        }

downloadFileFTP(uwind_dict)
downloadFileFTP(vwind_dict)

# =============================================================================
# Surface Air Temperature (SAT)
# =============================================================================

SAT_dict = {
        'filename': 'air.sig995.2019.nc',
        'host': 'ftp.cdc.noaa.gov',
        'location': '/Datasets/ncep.reanalysis.dailyavgs/surface/'
        }

for year_int in range(1948, 2019):
    year_str = str(year_int)
    SAT_dict['filename'] = 'air.sig995.%s.nc' % year_str
    downloadFileFTP(SAT_dict, outdir='sat')

SATmon_dict = {
        'filename': 'air.mon.mean.nc',
        'host': 'ftp.cdc.noaa.gov',
        'location': '/Datasets/ncep.reanalysis.derived/surface/'
        }

downloadFileFTP(SATmon_dict)

# =============================================================================
# indian ocean dipole  (IOD) index
# =============================================================================
IOD_dict = {
        'url': 'https://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Data/dmi.long.data',
        'filename': 'iod.txt'
        }

downloadFileHTTP(IOD_dict)

# =============================================================================
# =============================================================================
# # Postprocess
# =============================================================================
# =============================================================================
print_header("Postprocess Data")
from ninolearn.postprocess.prepare import prep_nino_seasonal, prep_nino_month, prep_wwv

prep_nino_seasonal()
prep_nino_month(index="3.4")
prep_nino_month(index="3")
prep_nino_month(index="1+2")
prep_nino_month(index="4")
prep_wwv()

from ninolearn.IO import read_raw
from ninolearn.postprocess.anomaly import postprocess
from ninolearn.postprocess.regrid import to2_5x2_5
# postprocess sst data from ERSSTv5
sst_ERSSTv5 = read_raw.sst_ERSSTv5()
postprocess(sst_ERSSTv5)

# postprocess data from NCEP/NCAR reanalysis
uwind = read_raw.uwind()
postprocess(uwind)

vwind = read_raw.vwind()
postprocess(vwind)

sat = read_raw.sat(mean='monthly')
postprocess(sat)

# postprocess sst date from HadISST date set
sst_HadISST = read_raw.sst_HadISST()
postprocess(sst_HadISST)

# postprocess sat daily values from NCEP/NCAR reanalysis
sat_daily = read_raw.sat(mean='daily')
postprocess(sat_daily)

# postprocess ssh values from ORAP5
ssh = read_raw.ssh()
postprocess(ssh)

#%% post process values from GODAS
ssh_godas = read_raw.ssh_godas()
ssh_godas_regrid = to2_5x2_5(ssh_godas)
postprocess(ssh_godas_regrid)
