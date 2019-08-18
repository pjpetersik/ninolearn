from ninolearn.IO.read_processed import data_reader

import matplotlib.pyplot as plt
plt.close("all")
reader = data_reader(startdate='1978-01', enddate='2018-12', lon_min=30)

wp_edge = reader.read_csv('wp_edge', processed='total')
wp_edge_dec = wp_edge.rolling(24, center=False).mean()

olr = reader.read_netcdf('olr', dataset='NCAR', processed='anom')
olr = olr.sortby('lat', ascending=False)
olr_cp = olr.loc[dict(lat=slice(2.5,-2.5), lon=slice(160, 180))].mean(dim='lat', skipna=True).mean(dim='lon', skipna=True)
olr_cp_dec = olr_cp.rolling(time=24, center=False).mean()

time = wp_edge.index

fig, ax1 = plt.subplots()
ax1.plot(time, - olr_cp_dec, 'r', label='Mean SST')

ax2 = ax1.twinx()
ax2.plot(time, wp_edge_dec, 'k')