import matplotlib.pyplot as plt
import pandas as pd

from ninolearn.IO.read_post import data_reader

reader = data_reader(startdate='1990-01', enddate='2018-01')

nwm = reader.read_network_metrics('air', dataset='NCEP', processed='deviation')
nino34 = reader.nino34_anom()
nino34max = nino34.abs().max()

nino_color = nino34.copy()
nino_color[abs(nino_color)<1.] = 0

plt.close("all")

plt.figure(1)
nwm['fraction_clusters_size_2'].plot()
#plt.legend()

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
        
        windowmax = abs(nino34.loc[start_date:end_date]).max()
        sign = nino34.loc[start_date]
        
        alpha = windowmax/nino34max * 0.8
        
        if sign>0:
            plt.axvspan(start_date, end_date, facecolor='red', alpha=alpha)
        
        else:
            plt.axvspan(start_date, end_date, facecolor='blue', alpha=alpha)

plt.figure(2)
nino34.plot()