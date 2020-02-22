"""
The downloaded data needed to be prepared to have the similiar time-axis.

All spatial data is regridded to the 2.5x2.5 grid of the NCEP
reanalysis data.

Some variables are computed, i.e the wind stress field, the wind speed and
the warm pool edge.
"""
import numpy as np

from ninolearn.utils import print_header
from ninolearn.preprocess.prepare import prep_oni, prep_nino_month, prep_wwv
from ninolearn.preprocess.prepare import prep_iod, prep_K_index, prep_wwv_proxy
from ninolearn.preprocess.prepare import calc_warm_pool_edge, prep_other_forecasts

print_header("Prepare Data")

# =============================================================================
# Prepare the incedes
# =============================================================================
prep_oni()
prep_nino_month(index="3.4")
prep_nino_month(index="3")
prep_nino_month(index="1+2")
prep_nino_month(index="4")
prep_wwv()
prep_wwv(cardinal_direction="west")
prep_iod()
prep_K_index()
prep_wwv_proxy()

# =============================================================================
# Prepare the gridded data
# =============================================================================
from ninolearn.IO import read_raw
from ninolearn.preprocess.anomaly import postprocess, saveAnomaly
from ninolearn.preprocess.regrid import to2_5x2_5

# postprocess sst data from ERSSTv5
sst_ERSSTv5 = read_raw.sst_ERSSTv5()
sst_ERSSTv5_regrid = to2_5x2_5(sst_ERSSTv5)
postprocess(sst_ERSSTv5_regrid)

# NCEP reanalysis
uwind = read_raw.uwind()
postprocess(uwind)

vwind = read_raw.vwind()
postprocess(vwind)

# post process values from ORAS4 ssh
ssh_oras4 = read_raw.oras4()
ssh_oras4_regrid = to2_5x2_5(ssh_oras4)
postprocess(ssh_oras4_regrid)

# OLR
olr_ncar = read_raw.olr()
olr_ncar_regrid = to2_5x2_5(olr_ncar)
postprocess(olr_ncar_regrid)

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

# =============================================================================
# Postprocessing based on already postprocessd data
# =============================================================================
calc_warm_pool_edge()

# =============================================================================
# Prepare the other forecasts
# =============================================================================
prep_other_forecasts()
