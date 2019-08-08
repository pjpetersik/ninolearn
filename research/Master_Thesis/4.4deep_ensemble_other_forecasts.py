from os.path import join
import xarray as xr
from ninolearn.private import plotdir
from ninolearn.pathes import postdir
import matplotlib.pyplot as plt
from ninolearn.IO.read_post import data_reader
from ninolearn.plot.prediction import plot_prediction
plt.close('all')

start = '2003-01'
end = '2017-12'
reader = data_reader(startdate=start, enddate=end)
oni = reader.read_csv('oni')

data = xr.open_dataset(join(postdir, f'DE_forecasts.nc'))
data_of = xr.open_dataset(join(postdir, f'other_forecasts.nc'))

lead = 6
lead_DE = lead//3

UU_DE_mean = data['UU DE mean'].loc[start:end][:,lead_DE]
UU_DE_std = data['UU DE std'].loc[start:end][:,lead_DE]

plt.subplots(figsize=(9,3))
plot_prediction(UU_DE_mean.target_season.values, UU_DE_mean.values,std=UU_DE_std.values,
                alpha=0.4
                )

target_season = data_of['target_season'].loc[start:end].to_pandas()

NASA_GMAO = data_of['NASA GMAO'].loc[start:end, lead]
NCEP_CFS = data_of['NCEP CFS'].loc[start:end, lead]
JMA = data_of['JMA'].loc[start:end, lead]
SCRIPPS =  data_of['SCRIPPS'].loc[start:end, lead]
ECMWF = data_of['ECMWF'].loc[start:end, lead]
KMA_SNU = data_of['KMA SNU'].loc[start:end, lead]

UBC_NNET = data_of['UBC NNET'].loc[start:end, lead]
UCLA_TCD = data_of['UCLA-TCD'].loc[start:end, lead]
CPC_MRKOV = data_of['CPC_MRKOV'].loc[start:end, lead]
CPC_CA = data_of['CPC_CA'].loc[start:end, lead]
CPC_CCA = data_of['CPC_CCA'].loc[start:end, lead]




alpha = 1
plt.plot(target_season, UBC_NNET, alpha=alpha, label='UBC NNET', ls='--')
plt.plot(target_season, UCLA_TCD, alpha=alpha, label='UCLA-TCD', ls='--')
plt.plot(target_season, CPC_MRKOV, alpha=alpha, label='CPC MRKOV', ls='--')
plt.plot(target_season, CPC_CA, alpha=alpha, label='CPC CA', ls='--')
plt.plot(target_season, CPC_CA, alpha=alpha, label='CPC CCA', ls='--')

plt.plot(target_season, NASA_GMAO, alpha=alpha, label='NASA GMAO')
plt.plot(target_season, NCEP_CFS, alpha=alpha, label='NCEP CFS')
plt.plot(target_season, JMA, alpha=alpha, label='JMA')
plt.plot(target_season, SCRIPPS, alpha=alpha, label='SCRIPPS')
plt.plot(target_season, ECMWF, alpha=alpha, label='ECMWF')
plt.plot(target_season, KMA_SNU, alpha=alpha, label='KMA SNU')

plt.plot(oni.index, oni, c='r', label='ONI', lw=2)

plt.xlim(UU_DE_mean.target_season.values[0], UU_DE_mean.target_season.values[-1])
plt.ylim(-3,3)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.axhspan(-0.5, -6, facecolor='blue',  alpha=0.1,zorder=0)
plt.axhspan(0.5, 6, facecolor='red', alpha=0.1,zorder=0)


plt.title(f"Lead time: {lead} month")
plt.grid()
plt.xlabel('Time [Year]')
plt.ylabel('ONI [K]')
plt.tight_layout()

plt.savefig(join(plotdir, f'compare_prediction_lead{lead}.pdf'))

