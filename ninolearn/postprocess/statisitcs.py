import xarray as xr
from os.path import join
import numpy as np

from ninolearn.IO import read_raw
from ninolearn.pathes import postdir


def computeDeviation(data, save=False):
    """
    remove the over all time mean from a time series
    """

    time_mean = data.mean(dim = 'time', skipna=True)
    deviation = data - time_mean
    
    deviation.name = ''.join([data.name, '.deviation'])
    
    deviation.attrs = data.attrs.copy()
    deviation.attrs['long_name'] = ''.join(['Deviation of ',deviation.attrs['long_name']])
    deviation.attrs['var_desc'] = ''.join([deviation.attrs['var_desc'], ' Deviation'])
    deviation.attrs['statistic'] = 'Substracted the Mean'
    deviation.attrs['actual_range'][0] = deviation.min()
    deviation.attrs['actual_range'][1] = deviation.max()
    del deviation.attrs['valid_range']
    
    if save:
        save_data(deviation)
    else:
        return deviation

def computeNormalized(data, save=False):
    """
    normalize the data
    """
    
    dev = computeDeviation(data)
    std = data.std(dim = 'time', skipna=True)
    norm = dev/std
    
    norm.name = ''.join([data.name, '.norm'])
    
    norm.attrs = data.attrs.copy()
    norm.attrs['long_name'] = ''.join(['Normalized ',norm.attrs['long_name']])
    norm.attrs['var_desc'] = ''.join(['Normalized ',norm.attrs['var_desc']])
    norm.attrs['statistic'] = 'Substracted the Mean. Divided by standerd Deviation.'
    norm.attrs['actual_range'][0] = norm.min()
    norm.attrs['actual_range'][1] = norm.max()
    del norm.attrs['valid_range']
    
    if save:
        save_data(norm)
    else:
        return norm
    
def save_data(data):
    data.to_netcdf(join(postdir,''.join([data.name,'.nc'])))
    

if __name__ == "__main__":
    
    data = read_raw.sst_ERSSTv5()
    save_data(data)
    
    computeDeviation(data,save=True)
    computeNormalized(data,save=True)
    dev = computeDeviation(data)
    norm = computeNormalized(data)
    
    