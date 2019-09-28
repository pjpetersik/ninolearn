"""
The downloaded data needed to be prepared to have the similiar time-axis.

All spatial data is regridded to the 2.5x2.5 grid of the NCEP
reanalysis data.

Some variables are computed, i.e the wind stress field, the wind speed and
the warm pool edge.
"""

from ninolearn.preprocess.prepare import prep_oni, prep_wwv
from ninolearn.preprocess.prepare import prep_iod, prep_K_index, prep_wwv_proxy

# =============================================================================
# Prepare the incedes
# =============================================================================
prep_oni()
prep_wwv()
prep_iod()
prep_K_index()
prep_wwv_proxy()

