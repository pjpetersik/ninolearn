import matplotlib.pyplot as plt
import pandas as pd

from ninolearn.IO.read_post import data_reader



def nino_background(nino_data, nino_treshold=0.8):
    nino_color = nino_data.copy()
    nino_color[abs(nino_color)<nino_treshold] = 0
    
    ninomax = nino_data.abs().max()
    start = True
    end = False
    
    for i in range(len(nino_color)):
        if nino_color[i]!=0 and start:
            start_date = nino_color.index[i]
            start = False
            end = True
        
        if nino_color[i]==0 and end:
            end_date = nino_color.index[i-1]
            start = True
            end = False
            
            windowmax = abs(nino_data.loc[start_date:end_date]).max()
            sign = nino34.loc[start_date]
            
            alpha = (windowmax - nino_treshold)/(ninomax-nino_treshold) * 0.8
            
            if sign>0:
                plt.axvspan(start_date, end_date, facecolor='red', alpha=alpha)
            else:
                plt.axvspan(start_date, end_date, facecolor='blue', alpha=alpha)
                
if __name__ == "__main__":
    reader = data_reader(startdate='1950-01', enddate='2018-12')
    
    nwm = reader.read_network_metrics('sst', dataset='ERSSTv5', processed='anom')
    nino34 = reader.nino34_anom()
    
    
    plt.close("all")
    
    plt.figure(figsize=(12,4))
    nwm['threshold'].plot(c='k')
    nino_background(nino34)
    
    #'global_transitivity'
    #'avelocal_transmissivity'
    #'fraction_clusters_size_2'
    #'fraction_clusters_size_3'
    #'fraction_clusters_size_5'
    #'fraction_giant_component'
    #'average_path_length'
    #'hamming_distance'
    #'corrected_hamming_distance'
    
    plt.figure(2)
    nino34.plot()