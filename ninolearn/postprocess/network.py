import igraph 
import numpy as np
import pandas as pd
from os.path import join

from ninolearn.IO.read_post import data_reader
from ninolearn.pathes import postdir

class climateNetwork(igraph.Graph):
    """
    Child object of the igraph.Graph class for the construction of a complex
    climate network
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
        adjacency[correalation_matrix>threshold] = 1.
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
    
    def hamming_distance(self,old_adjacency):
        """
        comput the Hamming distance
        :param other_graph: a igragph.Graph instance with which the Graph should be compared
        """
        try:
            N = self.vcount()
            H = np.abs(np.sum(self.adjacency_array - old_adjacency)) / (N * (N - 1))
            return H
        except:
            print("Wrong input for computation of hamming distance.")
            return 0
 
       
class networkMetricsSeries(object):
    def __init__(self, data_set='sst_ERSSTv5', processed='deviation', threshold= 0.99, startdate='1948-01', enddate='1948-12'):
        """
        Class for the computation of network metrics time series
        
        :type data_set: str
        :param data_set: the data_set that should be used to build the network
        
        :type processed: str
        :param processed: either '','deviation' or 'norm'
        
        :type threshold: float
        :param threshold: the threshold for a the correlation coeficent between 
        two grid point to be considered as connected
        
        :param startdate: start of the window for the computation of network metrics
        :param enddate: end of the window for the computation of network metrics
        """
        self.data_set = data_set
        self.processed = processed
        
        self.threshold = threshold
        
        self.startdate = startdate 
        self.enddate = enddate
        
        self.reader = data_reader(startdate=startdate, enddate=enddate)
        
        self.initalizeSeries()
        
    def initalizeSeries(self):
        """
        initializes the pandas Series and array that saves the adjacency of the 
        network from the previous time step
        """
        self.global_transitivity = pd.Series()
        self.avglocal_transitivity = pd.Series()
        self.frac_cluster_size2 = pd.Series()
        self.frac_cluster_size3 = pd.Series()
        self.frac_cluster_size5 = pd.Series()
        self.frac_giant = pd.Series()
        self.avg_path_length = pd.Series()
        self.hamming_distance = pd.Series()
        
        self._old_adjacency = np.array([])
        
    def computeCorrelationMatrix(self):
        data = eval(f'self.reader.{self.data_set}(processed="{self.processed}")')
        
        # Reshape
        data3Darr  = np.array(data)
        
        dims = data.coords.dims
        time_index = dims.index('time')
        lat_index = dims.index('lat')
        lon_index = dims.index('lon')
        
        len_time = data3Darr.shape[time_index]
        len_lat = data3Darr.shape[lat_index]
        len_lon = data3Darr.shape[lon_index]
        
        data2Darr = data3Darr.reshape(len_time,len_lat*len_lon)

        # Correlation matrix
        df2Darr = pd.DataFrame(data2Darr)
        df2Darr = df2Darr.dropna(axis=1)
        df_corrcoef = df2Darr.corr()
        
        corrcoef = df_corrcoef.to_numpy()
        return corrcoef
        
    def computeNetworkMetrics(self,corrcoef):
        """
        computes network metrics from a correlation matrix in combination with the
        already given threshold
        
        :param corrcoef: the correlation matrix
        """
        cn = climateNetwork.from_correalation_matrix(corrcoef,threshold=self.threshold)
        
        save_date = self.reader.enddate + pd.DateOffset(months=1)
        
        # C1 as in Newman (2003) and Eq. (6) in Radebach et al. (2013)
        self.global_transitivity[save_date] = cn.transitivity_undirected()
        
        # C2 as in Newman (2003) and Eq. (7) in Radebach et al. (2013)
        self.avglocal_transitivity[save_date] = cn.transitivity_avglocal_undirected(mode="zero")
        
        # fraction of nodes in clusters of size 2
        self.frac_cluster_size2[save_date] =  cn.cluster_fraction(2)
        
        # fraction of nodes in clusters of size 3
        self.frac_cluster_size3[save_date] =  cn.cluster_fraction(3)
        
        # fraction of nodes in clusters of size 3
        self.frac_cluster_size5[save_date] =  cn.cluster_fraction(5)
        
        # fraciont of nodes in giant component
        self.frac_giant[save_date] = cn.giant_fraction()
        
        # average path length
        self.avg_path_length[save_date] = cn.average_path_length()
        
        # hamming distance
        self.hamming_distance[save_date] = cn.hamming_distance(self._old_adjacency)        
        
        # copy the old adjacency
        self._old_adjacency = cn.adjacency_array.copy()
        
    def save(self):
        self.data = pd.DataFrame({'global_transitivity' : self.global_transitivity,
                       'avelocal_transmissivity': self.avglocal_transitivity,
                       'fraction_clusters_size_2': self.frac_cluster_size2, 
                       'fraction_clusters_size_3': self.frac_cluster_size3, 
                       'fraction_clusters_size_5': self.frac_cluster_size5, 
                       'fraction_giant_component': self.frac_giant,
                       'average_path_length':self.avg_path_length,
                       'hamming_distance':self.hamming_distance
                       })
    
        self.data.to_csv(join(postdir,'network_metrics.csv'))
    
    def computeTimeSeries(self):
        while self.reader.enddate != pd.to_datetime('2019-01-01'):
            print(f'{self.reader.startdate} till {self.reader.enddate}')
            
            corrcoef = self.computeCorrelationMatrix()
            self.computeNetworkMetrics(corrcoef)
            
            self.reader.shift_window()
        
        self.save()
            

if __name__ =="__main__":
    n = networkMetricsSeries()
    n.computeTimeSeries()