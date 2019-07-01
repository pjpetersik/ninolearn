import numpy as np

from ninolearn.download import downloadFileFTP, downloadFileHTTP, unzip_gz
from ninolearn.private import CMEMS_password, CMEMS_username
from ninolearn.utils import print_header

#%% =============================================================================
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
# ORAS4 data
# =============================================================================
ORAS4_dict = {'filename': 'zos_oras4_1m_1958_grid_1x1.nc',
              'host': 'ftp-icdc.cen.uni-hamburg.de',
              'location': '/EASYInit/ORA-S4/monthly_1x1/'
              }

for i in range(1958, 2018):
    ORAS4_dict['filename'] = f'zos_oras4_1m_{i}_grid_1x1.nc'
    downloadFileFTP(ORAS4_dict, outdir = 'ssh_oras4')

# =============================================================================
# GODAS data
# =============================================================================
GODAS_dict = {'filename': 'sshg.1980.nc',
              'host': 'ftp.cdc.noaa.gov',
              'location': '/Datasets/godas/'
              }

for i in range(1980,2019):
    #ssh
    GODAS_dict['filename'] = f'sshg.{i}.nc'
    downloadFileFTP(GODAS_dict, outdir = 'sshg_godas')

    #u-current
    GODAS_dict['filename'] = f'ucur.{i}.nc'
    downloadFileFTP(GODAS_dict, outdir = 'ucur_godas')

    #v-current
    GODAS_dict['filename'] = f'vcur.{i}.nc'
    downloadFileFTP(GODAS_dict, outdir = 'vcur_godas')

# =============================================================================
# WWV
# =============================================================================
WWV_dict = {
        'filename': 'wwv.dat',
        'url': 'https://www.pmel.noaa.gov/tao/wwv/data/wwv.dat'
        }

WWV_West_dict = {
        'filename': 'wwv_west.dat',
        'url': 'https://www.pmel.noaa.gov/tao/wwv/data/wwv_west.dat'
        }

downloadFileHTTP(WWV_dict)
downloadFileHTTP(WWV_West_dict)
#%% =============================================================================
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
# Ocean heat content
# =============================================================================
#https://www.ncdc.noaa.gov/cdr/oceanic/ocean-heat-content

HCA_dict={'filename': 'hca.nc',
          'url': 'http://data.nodc.noaa.gov/woa/DATA_ANALYSIS/3M_HEAT_CONTENT/NETCDF/heat_content/heat_content_anomaly_0-700_seasonal.nc'}

downloadFileHTTP(HCA_dict)

#%% =============================================================================
# OLR
# =============================================================================
OLRmon_dict = {
        'filename': 'olr.mon.mean.nc',
        'host': 'ftp.cdc.noaa.gov',
        'location': '/Datasets/interp_OLR/'
        }

downloadFileFTP(OLRmon_dict)
#%% =============================================================================
# =============================================================================
# # Postprocess
# =============================================================================
# =============================================================================
print_header("Postprocess Data")
from ninolearn.postprocess.prepare import prep_nino_seasonal, prep_nino_month, prep_wwv, prep_iod, prep_K_index, prep_wwv_proxy

prep_nino_seasonal()
prep_nino_month(index="3.4")
prep_nino_month(index="3")
prep_nino_month(index="1+2")
prep_nino_month(index="4")
prep_wwv()
prep_wwv(cardinal_direction="west")
prep_iod()
prep_K_index()
prep_wwv_proxy()


#%%
from ninolearn.IO import read_raw
from ninolearn.postprocess.anomaly import postprocess, saveAnomaly
from ninolearn.postprocess.regrid import to2_5x2_5
#%% postprocess sst data from ERSSTv5

sst_ERSSTv5 = read_raw.sst_ERSSTv5()
sst_ERSSTv5_regrid = to2_5x2_5(sst_ERSSTv5)
postprocess(sst_ERSSTv5_regrid)

# postprocess sat daily values from NCEP/NCAR reanalysis monthly
sat = read_raw.sat(mean='monthly')
postprocess(sat)

uwind = read_raw.uwind()
postprocess(uwind)

vwind = read_raw.vwind()
postprocess(vwind)

# post process values from ORAS4 ssh
ssh_oras4 = read_raw.oras4()
ssh_oras4_regrid = to2_5x2_5(ssh_oras4)
postprocess(ssh_oras4_regrid)

# post process values from GODAS
ssh_godas = read_raw.godas(variable='sshg')
ssh_godas_regrid = to2_5x2_5(ssh_godas)
postprocess(ssh_godas_regrid)

#%%TODO for postprecessing of ucur select one level at a time!
ucur_godas = read_raw.godas(variable='ucur')
ucur_godas_regrid = to2_5x2_5(ucur_godas)
postprocess(ucur_godas_regrid)

vcur_godas = read_raw.godas(variable='vcur')
vcur_godas_regrid = to2_5x2_5(vcur_godas)
postprocess(vcur_godas_regrid)

#%%
hca_ndoc = read_raw.hca_mon()
hca_ndoc_regrid = to2_5x2_5(hca_ndoc)
saveAnomaly(hca_ndoc_regrid, False, compute=False)

#%%
olr_ncar = read_raw.olr()
olr_ncar_regrid = to2_5x2_5(olr_ncar)
postprocess(olr_ncar_regrid)

#%%

# =============================================================================
# Calculate some variables
# =============================================================================
wspd = np.sqrt(uwind**2 + vwind**2)
wspd.attrs = uwind.attrs.copy()
wspd.name = 'wspd'
wspd.attrs['long_name'] = 'Monthly Mean Wind Speed at sigma level 0.995'
wspd.attrs['var_desc'] = 'wind-speed'
postprocess(wspd)

taux = uwind * wspd
taux.attrs = uwind.attrs.copy()
taux.name = 'taux'
taux.attrs['long_name'] = 'Monthly Mean Zonal Wind Stress at sigma level 0.995'
taux.attrs['var_desc'] = 'x-wind-stress'
taux.attrs['units'] = 'm^2/s^2'
postprocess(taux)

tauy = vwind * wspd
tauy.attrs = uwind.attrs.copy()
tauy.name = 'tauy'
tauy.attrs['long_name'] = 'Monthly Mean Meridional Wind Stress at sigma level 0.995'
tauy.attrs['var_desc'] = 'y-wind-stress'
tauy.attrs['units'] = 'm^2/s^2'
postprocess(tauy)


#%%
"""
ARCHIVED
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


SSH_GFDL_dict = {
        'filename': 'zos_Omon_GFDL-CM3_piControl_r1i1p1_000101-000512.nc',
        'host': 'nomads.gfdl.noaa.gov',
        'location': '/CMIP5/output1/NOAA-GFDL/GFDL-CM3/piControl/mon/ocean/Omon/r1i1p1/v20110601/zos/'
        }

for year_int in np.arange(1,500,5):
    year_start = f"{year_int:03d}"
    year_end = f"{year_int+4:03d}"

    SSH_GFDL_dict['filename'] = f'zos_Omon_GFDL-CM3_piControl_r1i1p1_0{year_start}01-0{year_end}12.nc'
    downloadFileFTP(SSH_GFDL_dict, outdir='ssh_gfdl')


#daily
sat_daily = read_raw.sat(mean='daily')
postprocess(sat_daily)

# postprocess ssh values from ORAP5
ssh = read_raw.ssh()
postprocess(ssh)

# postprocess sst date from HadISST date set
sst_HadISST = read_raw.sst_HadISST()
sst_HadISST_regrid = to2_5x2_5(sst_HadISST)
postprocess(sst_HadISST_regrid)

# postprocess data from NCEP/NCAR reanalysis
uwind = read_raw.uwind()
postprocess(uwind)

vwind = read_raw.vwind()
postprocess(vwind)


#%% post process values for GFDL control run
sat_gfdl = read_raw.sat_gfdl()
sat_gfdl_regrid = to2_5x2_5(sat_gfdl)
postprocess(sat_gfdl_regrid, ref_period=False)

sst_gfdl = read_raw.sst_gfdl()
sst_gfdl_regrid = to2_5x2_5(sst_gfdl)
postprocess(sst_gfdl_regrid, ref_period=False)

ssh_gfdl = read_raw.ssh_gfdl()
ssh_gfdl_regrid = to2_5x2_5(ssh_gfdl)
postprocess(ssh_gfdl_regrid, ref_period=False)

"""