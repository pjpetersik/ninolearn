import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ninolearn.IO.read_post import data_reader

reader = data_reader()

years = np.arange(1990,2018)
C = np.zeros(len(years))

for i in range(len(years)):
    print("read")
    reader.startdate = '%s-01'%years[i]
    reader.enddate = '%s-12'%years[i]
    
    data = reader.sst_ERSSTv5(processed='norm')
    
    #%% =============================================================================
    # Reshape
    # =============================================================================
    print("reshape")
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
    print("correlate")
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
    print("graph")
    graph = nx.from_numpy_array(adjacency)
    undirected_graph = graph.to_undirected()
    C[i] = nx.algorithms.cluster.average_clustering(undirected_graph)
    
    print(years[i])
    print(C[i])