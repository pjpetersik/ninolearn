import xarray as xr
from os.path import join, exists

from ninolearn.pathes import postdir
from ninolearn.postprocess.statistics import toPostDir

# =============================================================================
# Computation
# =============================================================================
def computeTimemean(data):  
    """
    Compute or read the timely mean of a data set. In case no data can be read, save
    the computed data.
    """
    path = join(postdir,''.join([data.name, '.mean','.nc']))
    if not exists(path):
        print("Compute time mean")
        try:
            timemean = data.loc['1948-01-01':'2018-12-31'].mean(dim = 'time_counter', skipna=True)
        except:
            timemean = data.loc['1948-01-01':'2018-12-31'].mean(dim = 'time', skipna=True)
        timemean = timemean.compute()
        timemean.to_netcdf(path)
    else:
        print("Read time mean")
        timemean = xr.open_dataarray(path)
    return timemean

def computeStd(data):
    """
    Compute or read the standard deviation of a data set. In case no data can be read, save
    the computed data.
    """
    path = join(postdir,''.join([data.name, '.std','.nc']))
    if not exists(path):
        print("Compute standard deviation")
        try:
            std = data.loc['1948-01-01':'2018-12-31'].mean(dim = 'time', skipna=True)
        except:
            std = data.loc['1948-01-01':'2018-12-31'].mean(dim = 'time_counter', skipna=True)
        std = std.compute()
        std.to_netcdf(path)
    else:
        print("Read time mean")
        std = xr.open_dataarray(path)
    return std

def computeDeviation(data):
    timemean = computeTimemean(data)
    deviation = data - timemean
    return deviation

def computeNormalized(data):
    timemean = computeTimemean(data)
    std = computeStd(data)
    norm = (data-timemean) / std
    return norm

# =============================================================================
# Saving
# =============================================================================

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
        deviation.attrs['statistic'] = 'Substracted the Mean'
       
        try:
            del deviation.attrs['valid_range']
        except:
            pass
        
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
        norm.attrs['statistic'] = 'Substracted the Mean. Divided by standerd Deviation.'
        
        try:
            del norm.attrs['valid_range']
        except:
            pass
        norm.to_netcdf(path)

def postprocessDask(data,new=False):
    """
    combine all the postprocessing functions in one data routine
    :param data: xarray data array
    :param new: compute the statistics again (default = False)
    """
    toPostDir(data)
    saveDeviation(data, new)
    saveNormalized(data, new)
    
if __name__=="__main__": 
    from ninolearn.IO import read_raw
    
    print("Read")
    data = read_raw.sat(mean='daily')
    postprocessDask(data)





