# -*- coding: utf-8 -*-

from ninolearn.IO.read_post import data_reader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

plt.close("all")
reader = data_reader(startdate='1980-02')


nino34 = reader.read_csv('nino3.4S')
max_lag = 13

auto_corr = np.zeros((12, max_lag))
p_value = np.zeros((12, max_lag))
seas_ticks = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
                        'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']

for i in range(12):
    for j in range(max_lag):
        try:
            auto_corr[(i+j)%12,j],p_value[(i+j)%12,j] = pearsonr(nino34[i::12], nino34[i+j::12])
        except:
            auto_corr[(i+j)%12,j],p_value[(i+j)%12,j]  = pearsonr(nino34[i::12][:-1], nino34[i+j::12])

levels = np.linspace(-1, 1, 20+1)
fig, ax = plt.subplots(figsize=(5,3.5))

m = np.arange(1,13)
lag_arr = np.arange(max_lag)


C=ax.contourf(m,lag_arr,auto_corr.T, cmap=plt.cm.seismic,vmin=-1,vmax=1,levels=levels)
ax.set_xticks(m)
ax.set_xticklabels(seas_ticks, rotation='vertical')
ax.set_xlabel('Target Season')
ax.set_ylabel('Lag Month')
plt.colorbar(C, ticks=np.arange(-1,1.1,0.2))
plt.tight_layout()
ax.contour(m,lag_arr, p_value.T, levels=[0.01, 0.05, 0.1], linestyles=['solid', 'dashed', 'dotted'], colors='k')
