import igraph 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join

from ninolearn.IO.read_post import data_reader
from ninolearn.pathes import postdir
#%%



def computeNetworkMetrics():
    """
    routine to compute complex network metrics
    """
    reader = data_reader(startdate='1948-01', enddate='1948-12')
    C = pd.Series()

    while reader.enddate != pd.to_datetime('2019-01-01'):
        
        data = reader.sst_ERSSTv5(processed='norm')
        
        #%% =============================================================================
        # Reshape
        # =============================================================================
       
        data3Darr  = np.array(data)
        
        dims = data.coords.dims
        time_index = dims.index('time')
        lat_index = dims.index('lat')
        lon_index = dims.index('lon')
        
        len_time = data3Darr.shape[time_index]
        len_lat = data3Darr.shape[lat_index]
        len_lon = data3Darr.shape[lon_index]
        
        data2Darr = data3Darr.reshape(len_time,len_lat*len_lon)
        
        # =============================================================================
        # Correlation matrix
        # =============================================================================
        
        df = pd.DataFrame(data2Darr)
        
        df_corrcoef = df.corr()
        df_corrcoef = df_corrcoef.fillna(0)
        
        corrcoef = df_corrcoef.to_numpy()
        
        # =============================================================================
        # Adjacency
        # =============================================================================
        adjacency = np.zeros_like(corrcoef)
        adjacency[corrcoef>0.9] = 1.
        np.fill_diagonal(adjacency,0)
        
        # =============================================================================
        # Graph
        # =============================================================================
        
        G_igraph = igraph.Graph.Adjacency((adjacency > 0).tolist())
        
        C[reader.enddate + pd.DateOffset(months=1)]  = G_igraph.transitivity_undirected()
        
        
        print(f'{reader.startdate} till {reader.enddate}')
        print(C.loc[reader.enddate + pd.DateOffset(months=1)])
        reader.shift_window()
        
    df = pd.DataFrame({'clustering_coefficent':C})
    df.to_csv(join(postdir,'cc.csv'))
    
    
if __name__ =="__main__":
    reader = data_reader()
    nino34 = reader.nino34_anom()
    