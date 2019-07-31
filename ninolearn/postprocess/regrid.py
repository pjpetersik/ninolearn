from ninolearn.IO import read_raw

#import ESMF
import xarray as xr
import numpy as np
import xesmf as xe


def to2_5x2_5(data):
    """
    Regrids data the 2.5x2.5 from the NCEP reanalysis data set.

    :param data: An xarray dataArray or DataSet with with dimensions named\
    'lat' and 'lon'.

    """
    ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 90.01, 2.5)),
                     'lon': (['lon'], np.arange(0, 359.99, 2.5)),
                    }
                   )

    regridder = xe.Regridder(data, ds_out, 'bilinear')
    regrid_data = regridder(data)
    regrid_data.attrs = data.attrs
    return regrid_data
