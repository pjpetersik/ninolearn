# -*- coding: utf-8 -*-

from ninolearn.IO.read_processed import data_reader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from ninolearn.private import plotdir
from ninolearn.plot.evaluation import newcmp
from os.path import join

plt.close("all")
reader = data_reader(startdate='1962-01', enddate='2017-12')


oni = reader.read_csv('oni')
max_lag = 16

auto_corr = np.zeros((12, max_lag))
p_value = np.zeros((12, max_lag))
seas_ticks = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
                        'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']

for i in range(12):
    for j in range(max_lag):
        try:
            auto_corr[(i+j)%12,j],p_value[(i+j)%12,j] = pearsonr(oni[i::12], oni[i+j::12])
        except:
            try:
                auto_corr[(i+j)%12,j],p_value[(i+j)%12,j]  = pearsonr(oni[i::12][:-1], oni[i+j::12])
            except:
                auto_corr[(i+j)%12,j],p_value[(i+j)%12,j]  = pearsonr(oni[i::12][:-2], oni[i+j::12])

levels = np.linspace(0, 1, 20+1)
fig, ax = plt.subplots(figsize=(5,3.5))

p_value[auto_corr<=0] = 1

m = np.arange(1,13)
lag_arr = np.arange(max_lag) - 3


C=ax.contourf(m,lag_arr,auto_corr.T, cmap=newcmp,vmin=0,vmax=1,levels=levels, extend='min')
ax.set_xticks(m)
ax.set_xticklabels(seas_ticks, rotation='vertical')
ax.set_xlabel('Target Season')
ax.set_ylabel('Lead Time [Months]')
plt.colorbar(C, ticks=np.arange(-1,1.1,0.2))
plt.tight_layout()
ax.contour(m,lag_arr, p_value.T, levels=[0.01, 0.05, 0.1], linestyles=['solid', 'dashed', 'dotted'], colors='k')
plt.savefig(join(plotdir, 'autocorr.pdf'), dpi=360)