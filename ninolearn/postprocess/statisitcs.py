from os.path import join, exists

from ninolearn.IO import read_raw
from ninolearn.pathes import postdir

# =============================================================================
# Computation
# =============================================================================
def computeDeviation(data):
    """
    remove the over all time mean from a time series
    """
    time_mean = data.mean(dim = 'time', skipna=True)
    deviation = data - time_mean
    return deviation

def computeNormalized(data):
    """
    normalize the data
    """
    time_mean = data.mean(dim = 'time', skipna=True)
    deviation = data - time_mean
    std = deviation.std(dim = 'time', skipna=True)
    norm = deviation/std
    return norm


# =============================================================================
# Saving
# =============================================================================

def toPostDir(data):
    """
    save the basic data to the postdir
    """
    path = join(postdir,''.join([data.name,'.nc']))
    if exists(path):
         print (f"{data.name} already saved in post directory")
    else:
        print (f"save {data.name} in post directory")
        data.to_netcdf(path)
        
def saveDeviation(data, new):
    """
    save deviation to postdir
    """
    path = join(postdir,''.join([data.name, '.deviation','.nc']))
    
    if exists(path) and not new:
        print (f"{data.name} deviation already computed")
    
    else:
        print (f"Compute {data.name} deviation")
        
        deviation = computeDeviation(data)
        
        deviation.name = ''.join([data.name, '.deviation'])
        
        deviation.attrs = data.attrs.copy()
        deviation.attrs['long_name'] = ''.join(['Deviation of ',deviation.attrs['long_name']])
        deviation.attrs['var_desc'] = ''.join([deviation.attrs['var_desc'], ' Deviation'])
        deviation.attrs['statistic'] = 'Substracted the Mean'
        deviation.attrs['actual_range'][0] = deviation.min()
        deviation.attrs['actual_range'][1] = deviation.max()
        del deviation.attrs['valid_range']
        
        deviation.to_netcdf(path)

def saveNormalized(data, new):
    """
    save deviation to postdir
    """
    path = join(postdir,''.join([''.join([data.name, '.norm']),'.nc']))
    
    if exists(path) and not new:
        print (f"{data.name} normalized already computed")
    
    else:
        print (f"Compute {data.name} normalized")
        norm = computeNormalized(data)
        
        norm.name = ''.join([data.name, '.norm'])
        
        norm.attrs = data.attrs.copy()
        norm.attrs['long_name'] = ''.join(['Normalized ',norm.attrs['long_name']])
        norm.attrs['var_desc'] = ''.join(['Normalized ',norm.attrs['var_desc']])
        norm.attrs['statistic'] = 'Substracted the Mean. Divided by standerd Deviation.'
        norm.attrs['actual_range'][0] = norm.min()
        norm.attrs['actual_range'][1] = norm.max()
        del norm.attrs['valid_range']
        
        norm.to_netcdf(path)

def postprocess(data,new=False):
    """
    combine all the postprocessing functions in one data routine
    :param data: xarray data array
    :param new: compute the statistics again (default = False)
    """
    toPostDir(data)
    saveDeviation(data, new)
    saveNormalized(data, new)

if __name__ == "__main__":
    data = read_raw.sst_ERSSTv5()
    postprocess(data)