import matplotlib.pyplot as plt
from ninolearn.IO.read_post import data_reader
from ninolearn.IO import read_raw
from ninolearn.plot.nino_timeseries import nino_background
from ninolearn.utils import scale
from statsmodels.tsa.stattools import ccf
from scipy.stats import spearmanr
from mpl_toolkits.basemap import Basemap


import numpy as np
import pandas as pd

from ninolearn.postprocess.pca import pca

plt.close("all")

reader = data_reader(startdate='1955-02', enddate='2017-12')

pca_dechca = reader.read_statistic('pca', variable='dec_hca', dataset='NODC', processed='anom')
pca_decsst = reader.read_statistic('pca', variable='dec_sst', dataset='ERSSTv5', processed='anom')


plt.plot(-scale(pca_dechca['pca1']))
plt.plot(scale(pca_decsst['pca1']))
