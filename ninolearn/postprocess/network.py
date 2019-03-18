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
        """
        generate an igraph network form a adjacency matrix
        """
        np.fill_diagonal(adjacency,0)
        cls.adjacency_array = adjacency
        A = (adjacency > 0).tolist()
        
        # mode = 1 means undirected
        return cls.Adjacency(A,mode=1)
    
    @classmethod
    def from_correalation_matrix(cls,correalation_matrix,threshold = 0.9):
        """
        generate an igraph network from a correlation matrix
        """
        adjacency = np.zeros_like(correalation_matrix)
        adjacency[corrcoef>threshold] = 1.
        cls.threshold = threshold
        return cls.from_adjacency(adjacency)
    
    def giant_fraction(self):
        """
        returns the fraction of the nodes that are part of the giant component
        """
        nodes_total = self.vcount()
        nodes_giant = self.clusters().giant().vcount()
        return nodes_giant/nodes_total
        
    def cluster_fraction(self,size=2):
        """
        returns the fraction of the nodes that are part of a cluster of the given
        size (default: size=2)
        """
        nodes_total = self.vcount()
        nodes_cluster = self.clusters().sizes().count(size)
        return nodes_cluster/nodes_total


if __name__ == "__main__":
    reader = data_reader(startdate='1990-01', enddate='1991-12')
    
    global_transitivity = pd.Series()
    avglocal_transitivity = pd.Series()
    frac_cluster_size2 = pd.Series()
    frac_giant = pd.Series()
    
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
        
        save_date = reader.enddate + pd.DateOffset(months=1)
        
        # C1 as in Newman (2003) and Eq. (6) in Radebach et al. (2013)
        global_transitivity[save_date] = cn.transitivity_undirected()
        
        # C2 as in Newman (2003) and Eq. (7) in Radebach et al. (2013)
        avglocal_transitivity[save_date] = cn.transitivity_avglocal_undirected(mode="zero")
        
        # fraction of nodes in clusters of size 2
        frac_cluster_size2[save_date] =  cn.cluster_fraction(2)
        
        # fraciont of nodes in giant component
        frac_giant[save_date] = cn.giant_fraction()
        
        print(f'{reader.startdate} till {reader.enddate}')
        print(frac_cluster_size2.loc[reader.enddate + pd.DateOffset(months=1)])
        reader.shift_window()
        
    df = pd.DataFrame({'global transitivity' : global_transitivity,
                       'avelocal_transmissivity': avglocal_transitivity,
                       'fraction_clusters_size_2': frac_cluster_size2, 
                       'fraction_giant_component': frac_giant})
    
    df.to_csv(join(postdir,'network_metrics.csv'))
    

    import matplotlib.pyplot as plt
    reader = data_reader()
    nino = reader.nino34_anom()
    
    def normalize(data):
        return (data - data.mean())/data.std()
    
    plt.plot(normalize(nino))
    plt.plot(normalize(df))