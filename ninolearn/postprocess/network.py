import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ninolearn.IO.read import data_reader

reader = data_reader(startdate='1997-01', enddate='1997-12')

data = reader.sst_ERSSTv5()

# =============================================================================
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

graph = nx.from_numpy_array(adjacency)
clustering = nx.algorithms.cluster.average_clustering(graph)