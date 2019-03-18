"""
In this module, I want to write the code for the computation of complex climate 
network metrics
"""


import igraph 
import numpy as np
import pandas as pd
from os.path import join

from ninolearn.IO.read_post import data_reader
from ninolearn.pathes import postdir

class climateNetwork(igraph.Graph):
    """
    Wrapper object for the construction of a complex climate network
    """
    @classmethod
    def from_adjacency(cls,adjacency):
        np.fill_diagonal(adjacency,0)
        A = (adjacency > 0).tolist()
        return cls.Adjacency(A)
    
    @classmethod
    def from_correalation_matrix(cls,correalation_matrix,threshold = 0.9):
        adjacency = np.zeros_like(correalation_matrix)
        adjacency[corrcoef>threshold] = 1.
        return cls.from_adjacency(adjacency)
    

if __name__ == "__main__":
    reader = data_reader(startdate='1993-01', enddate='1993-12')
    
    CC = pd.Series()
    c2 = pd.Series()
    S =pd.Series()
    
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
        df = df.dropna(axis=1)
        df_corrcoef = df.corr()
        
        corrcoef = df_corrcoef.to_numpy()
        
        # =============================================================================
        # Cimate Network Graph
        # =============================================================================
        cn = climateNetwork.from_correalation_matrix(corrcoef,threshold=0.99)
        n = cn.vcount()
        
        clusters = cn.clusters()
        giant = clusters.giant()
        gn = giant.vcount()
        
        n2 = clusters.sizes().count(2)
        
        
        save_date = reader.enddate + pd.DateOffset(months=1)
        CC[save_date] = cn.transitivity_undirected()
        c2[save_date] =  n2/n
        S[save_date] = gn/n
        
        print(f'{reader.startdate} till {reader.enddate}')
        print(c2.loc[reader.enddate + pd.DateOffset(months=1)])
        reader.shift_window()
        
    df = pd.DataFrame({'clustering_coefficent' : CC,'c2':c2, 'S':S})
    df.to_csv(join(postdir,'network_metrics.csv'))
    
#%%
import matplotlib.pyplot as plt
reader = data_reader()
nino = reader.nino34_anom()

def normalize(data):
    return (data - data.mean())/data.std()

plt.plot(normalize(nino))
plt.plot(normalize(df))