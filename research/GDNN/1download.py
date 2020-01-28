"""
The following script downloads all data that was relevant for my master thesis.
"""

from ninolearn.download import download, sources
from ninolearn.utils import print_header

print_header("Download Data")

#%%
# =============================================================================
# Single files
# =============================================================================
download(sources.SST_ERSSTv5)
download(sources.ONI)
download(sources.NINOindices)
download(sources.IOD)
download(sources.OLR_NOAA)
download(sources.WWV)
download(sources.WWV_West)
download(sources.UWIND_NCEP)
download(sources.VWIND_NCEP)
download(sources.otherForecasts)

# =============================================================================
# Multiple files
# =============================================================================
for i in range(1958, 2018):
    sources.ORAS4['filename'] = f'zos_oras4_1m_{i}_grid_1x1.nc'
    download(sources.ORAS4, outdir = 'ssh_oras4')

for i in range(1980, 2019):
    #ssh
    sources.GODAS['filename'] = f'sshg.{i}.nc'
    download(sources.GODAS, outdir = 'sshg_godas')

    #u-current
    sources.GODAS['filename'] = f'ucur.{i}.nc'
    download(sources.GODAS, outdir = 'ucur_godas')

    #v-current
    sources.GODAS['filename'] = f'vcur.{i}.nc'
    download(sources.GODAS, outdir = 'vcur_godas')

for year_int in range(1948, 2019):
    year_str = str(year_int)
    sources.SAT_daily_NCEP['filename'] = 'air.sig995.%s.nc' % year_str
    download(sources.SAT_daily_NCEP, outdir='sat')