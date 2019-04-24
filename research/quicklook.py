import matplotlib.pyplot as plt
from ninolearn.IO.read_post import data_reader
from ninolearn.plot.nino_timeseries import nino_background
from statsmodels.tsa.stattools import ccf

plt.close("all")

reader = data_reader(startdate='1980-9', enddate='2008-01')
iod = reader.read_csv('iod')
nino34 = reader.read_csv('nino3.4M')

plt.subplots()
iod.plot()


iod_3m = iod.resample('3MS').mean()
iod_9 = iod_3m.loc[iod_3m.index.month==9]

iod_fill = iod_9.resample('MS').ffill()

plt.subplots()
#plt.xcorr(nino34.loc[:'2007-09-01'], iod_fill, maxlags=60)
plt.xcorr(nino34, iod, maxlags=300)