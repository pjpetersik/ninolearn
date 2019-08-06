import matplotlib.pyplot as plt
from ninolearn.IO.read_post import data_reader
from ninolearn.private import plotdir

from os.path import join

reader = data_reader(startdate='1950-02', enddate='2018-12', lon_min=30)
oni = reader.read_csv('oni')
iod = reader.read_csv('iod')
iod = iod.rolling(window=3, center=False).mean()

plt.close("all")
plt.subplots(figsize=(8,3))
plt.fill_between(oni.index, oni, 0, label="ONI")
plt.plot(iod, 'k', label="DMI")
plt.legend(loc=2)

plt.xlim(iod.index[0], iod.index[-1])
plt.ylim(-2.5,2.5)
plt.xlabel("Time [Year]")
plt.ylabel('Index Value')
plt.tight_layout()

plt.savefig(join(plotdir, 'iod_oni.pdf'))