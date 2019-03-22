from os.path import join, exists
from os import remove
import xarray as xr

from ninolearn.pathes import postdir
from ninolearn.utils import generateFileName, small_print_header
"""
TODO: Remove seasonality
"""
# =============================================================================
# # ===========================================================================
# # Computation
# # ===========================================================================
# =============================================================================

def computeTimemean(data):  
    """
    Compute or read the timely mean of a data set. In case no data can be read, save
    the computed data.
    """
    filename = generateFileName(data.name, dataset=data.dataset, processed='timemean',suffix='nc')
    path = join(postdir,filename)
    if not exists(path):
        print(f"- Compute {data.name} time mean")
        timemean = data.loc['1948-01-01':'2018-12-31'].mean(dim = 'time', skipna=True)
        timemean.to_netcdf(path)
    else:
        print(f"- Read {data.name} time mean")
        timemean = xr.open_dataarray(path)
    return timemean

def computeStd(data):
    """
    Compute or read the standard deviation of a data set. In case no data can be read, save
    the computed data.
    """
    filename = generateFileName(data.name, dataset=data.dataset, processed='std',suffix='nc')
    path = join(postdir,filename)
    if not exists(path):
        print(f"- Compute {data.name} standard deviation")
        std = data.loc['1948-01-01':'2018-12-31'].mean(dim = 'time', skipna=True)
        std.to_netcdf(path)
    else:
        print(f"- Read {data.name} time mean")
        std = xr.open_dataarray(path)
    return std

def computeDeviation(data):
    """
    remove the over all time mean from a time series
    """
    time_mean = computeTimemean(data)
    deviation = data - time_mean
    return deviation

def computeNormalized(data):
    """
    normalize the data
    """
    timemean = computeTimemean(data)
    std = computeStd(data)
    norm = (data-timemean) / std
    return norm


# =============================================================================
# =============================================================================
# # Attribute manipulation
# =============================================================================
# =============================================================================

def _delete_some_attributes(attrs):
    """
    delete some attributes from the orginal data set that lose meaning after data 
    processing
    
    :param attrs: the attribute list
    """
    to_delete_attrs = ['actual_range','valid_range']
    for del_attrs in to_delete_attrs:
        if del_attrs in  attrs:
            del attrs[del_attrs]
    return attrs

# =============================================================================
# # ===========================================================================
# # Saving
# # ===========================================================================
# =============================================================================
    
def toPostDir(data):
    """
    save the basic data to the postdir
    """
    filename = generateFileName(data.name, dataset=data.dataset,suffix='nc')
    path = join(postdir,filename)
    
    if exists(path):
         print (f"{data.name} already saved in post directory")
    else:
        print (f"save {data.name} in post directory")
        data.to_netcdf(path)
        
def saveDeviation(data, new):
    """
    save deviation to postdir
    """
    filename = generateFileName(data.name, dataset=data.dataset, processed='deviation',suffix='nc')
    path = join(postdir,filename)
    
    if exists(path) and not new:
        print (f"{data.name} deviation already computed")
    
    else:
        print (f"Compute {data.name} deviation")
        
        deviation = computeDeviation(data)
        
        deviation.name = ''.join([data.name, 'Deviation'])
        
        deviation.attrs = data.attrs.copy()
        deviation.attrs['statistic'] = 'Substracted the Mean'
        
        deviation.attrs = _delete_some_attributes(deviation.attrs)
        
        deviation.to_netcdf(path)


def saveNormalized(data, new):
    """
    save deviation to postdir
    """
    filename = generateFileName(data.name, dataset=data.dataset, processed='norm',suffix='nc')
    path = join(postdir,filename)
    
    
    if exists(path) and not new:
        print (f"{data.name} normalized already computed")
    
    else:
        print (f"Compute {data.name} normalized")
        norm = computeNormalized(data)
        
        norm.name = ''.join([data.name, 'Norm'])
        
        norm.attrs = data.attrs.copy()
        norm.attrs['statistic'] = 'Substracted the Mean. Divided by standerd Deviation.'
        
        norm.attrs = _delete_some_attributes(norm.attrs)
        
        norm.to_netcdf(path)



def postprocess(data,new=False):
    """
    combine all the postprocessing functions in one data routine
    :param data: xarray data array
    :param new: compute the statistics again (default = False)
    """
    small_print_header(f"Process {data.name} from {data.dataset}")
    toPostDir(data)
    saveDeviation(data, new)
    saveNormalized(data, new)