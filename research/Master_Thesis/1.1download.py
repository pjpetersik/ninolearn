"""
The following script downloads all data that was relevant for my master thesis.
"""

from ninolearn.download import download
from ninolearn.utils import print_header

from ninolearn.sources import SST_ERSSTv5, ONI, NINOindeces, ORAS4, GODAS, WWV, WWV_West
from ninolearn.sources import UWIND_NCEP, VWIND_NCEP, SAT_monthly_NCEP, SAT_daily_NCEP
from ninolearn.sources import IOD, HCA, OLR_NOAA


print_header("Download Data")

# =============================================================================
# Single files
# =============================================================================
download(SST_ERSSTv5)
download(ONI)
download(NINOindeces)
download(IOD)
download(HCA)
download(OLR_NOAA)
download(WWV)
download(WWV_West)
download(UWIND_NCEP)
download(VWIND_NCEP)
download(VWIND_NCEP)
download(SAT_monthly_NCEP)

# =============================================================================
# Multiple files
# =============================================================================
for i in range(1958, 2018):
    ORAS4['filename'] = f'zos_oras4_1m_{i}_grid_1x1.nc'
    download(ORAS4, outdir = 'ssh_oras4')

for i in range(1980,2019):
    #ssh
    GODAS['filename'] = f'sshg.{i}.nc'
    download(GODAS, outdir = 'sshg_godas')

    #u-current
    GODAS['filename'] = f'ucur.{i}.nc'
    download(GODAS, outdir = 'ucur_godas')

    #v-current
    GODAS['filename'] = f'vcur.{i}.nc'
    download(GODAS, outdir = 'vcur_godas')

for year_int in range(1948, 2019):
    year_str = str(year_int)
    SAT_daily_NCEP['filename'] = 'air.sig995.%s.nc' % year_str
    download(SAT_daily_NCEP, outdir='sat')