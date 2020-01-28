from os.path import join
import xarray as xr
import numpy as np
from ninolearn.pathes import processeddir
from ninolearn.private import plotdir
import matplotlib.pyplot as plt
from ninolearn.IO.read_processed import data_reader
from ninolearn.learn.skillMeasures import mean_srmse
from scipy.stats import pearsonr
from matplotlib.ticker import MaxNLocator

plt.close('all')

start = '2003-01'
end = '2017-12'
reader = data_reader(startdate=start, enddate=end)
oni = reader.read_csv('oni')

data = xr.open_dataset(join(processeddir, f'dem_forecasts.nc'))
data_of = xr.open_dataset(join(processeddir, f'other_forecasts.nc'))

lead_arr_of = np.arange(9)
lead_arr_DE = np.array([0,3,6,9])

corr_UBC_NNET = np.zeros(9)
corr_ECMWF = np.zeros(9)
corr_CFS = np.zeros(9)
corr_UCLA_TCD = np.zeros(9)
corr_JMA = np.zeros(9)
corr_NASA_GMAO = np.zeros(9)
corr_CPC_MRKOV = np.zeros(9)
corr_CPC_CA = np.zeros(9)
corr_CPC_CCA = np.zeros(9)
corr_NCEP_CFS = np.zeros(9)
corr_SCRIPPS = np.zeros(9)
corr_KMA_SNU = np.zeros(9)

ssrmse_UBC_NNET = np.zeros(9)
ssrmse_ECMWF = np.zeros(9)
ssrmse_CFS = np.zeros(9)
ssrmse_UCLA_TCD = np.zeros(9)
ssrmse_JMA = np.zeros(9)
ssrmse_NASA_GMAO = np.zeros(9)
ssrmse_CPC_MRKOV = np.zeros(9)
ssrmse_CPC_CA = np.zeros(9)
ssrmse_CPC_CCA = np.zeros(9)
ssrmse_NCEP_CFS = np.zeros(9)
ssrmse_SCRIPPS = np.zeros(9)
ssrmse_KMA_SNU = np.zeros(9)

def corr(data):
    nans = np.isnan(data)
    n_nans = len(data[np.isnan(data)])
    if n_nans<36:
        corr, _ = pearsonr(oni[~nans], data[~nans])
    else:
        corr=np.nan
    return corr

def ssrmse(data):
    nans = np.isnan(data)
    n_nans = len(data[np.isnan(data)])
    if n_nans<36:
        ssrmse = mean_srmse(oni[~nans], data[~nans], oni[~nans].index)
    else:
        ssrmse=np.nan
    return ssrmse


for i in range(9):
# =============================================================================
# Correaltion skills
# =============================================================================
    NASA_GMAO = data_of['NASA GMAO'].loc[start:end, i]
    NCEP_CFS = data_of['NCEP CFS'].loc[start:end, i]
    JMA = data_of['JMA'].loc[start:end, i]
    SCRIPPS =  data_of['SCRIPPS'].loc[start:end, i]
    ECMWF = data_of['ECMWF'].loc[start:end, i]
    KMA_SNU = data_of['KMA SNU'].loc[start:end, i]

    UBC_NNET = data_of['UBC NNET'].loc[start:end, i]
    UCLA_TCD = data_of['UCLA-TCD'].loc[start:end, i]
    CPC_MRKOV = data_of['CPC MRKOV'].loc[start:end, i]
    CPC_CA = data_of['CPC CA'].loc[start:end, i]
    CPC_CCA = data_of['CPC CCA'].loc[start:end, i]

    corr_NASA_GMAO[i] = corr(NASA_GMAO)
    corr_NCEP_CFS[i] = corr(NCEP_CFS)
    corr_JMA[i] = corr(JMA)
    corr_SCRIPPS[i] = corr(SCRIPPS)
    corr_KMA_SNU[i] = corr(KMA_SNU)
    corr_ECMWF[i] = corr(ECMWF)

    corr_UBC_NNET[i] = corr(UBC_NNET)
    corr_UCLA_TCD[i] = corr(UCLA_TCD)
    corr_CPC_MRKOV[i] = corr(CPC_MRKOV)
    corr_CPC_CA[i] = corr(CPC_CA)
    corr_CPC_CCA[i] = corr(CPC_CCA)
# =============================================================================
# SSRMSE
# =============================================================================
    ssrmse_NASA_GMAO[i] = ssrmse(NASA_GMAO)
    ssrmse_NCEP_CFS[i] = ssrmse(NCEP_CFS)
    ssrmse_JMA[i] = ssrmse(JMA)
    ssrmse_SCRIPPS[i] = ssrmse(SCRIPPS)
    ssrmse_KMA_SNU[i] = ssrmse(KMA_SNU)
    ssrmse_ECMWF[i] = ssrmse(ECMWF)

    ssrmse_UBC_NNET[i] = ssrmse(UBC_NNET)
    ssrmse_UCLA_TCD[i] = ssrmse(UCLA_TCD)
    ssrmse_CPC_MRKOV[i] = ssrmse(CPC_MRKOV)
    ssrmse_CPC_CA[i] = ssrmse(CPC_CA)
    ssrmse_CPC_CCA[i] = ssrmse(CPC_CCA)

corr_DE = np.zeros(4)
ssrmse_DE = np.zeros(4)

for i in range(4):
    UU_DE_mean = data['mean'].loc[start:end][:, i]
    corr_DE[i]=  corr(UU_DE_mean)
    ssrmse_DE[i]=  ssrmse(UU_DE_mean)


#%% =============================================================================
# Plot
# =============================================================================
plt.close("all")
ax = plt.figure(figsize=(6,3)).gca()

plt.plot(lead_arr_of, corr_UBC_NNET, label='UBC NNET', ls='--')
plt.plot(lead_arr_of, corr_UCLA_TCD, label='UCLA-TCD',  ls='--')
plt.plot(lead_arr_of, corr_CPC_MRKOV, label='CPC MRKOV', ls='--')
plt.plot(lead_arr_of, corr_CPC_CA, label='CPC CA', ls='--')
plt.plot(lead_arr_of, corr_CPC_CCA, label='CPC CCA', ls='--')

plt.plot(lead_arr_of, corr_NASA_GMAO, label='NASA GMAO')
plt.plot(lead_arr_of, corr_NCEP_CFS, label='NCEP CFS')
plt.plot(lead_arr_of, corr_JMA, label='JMA')
plt.plot(lead_arr_of, corr_SCRIPPS, label='SCRIPPS')
plt.plot(lead_arr_of, corr_ECMWF, label='ECMWF')
plt.plot(lead_arr_of, corr_KMA_SNU, label='KMA SNU')

plt.plot(lead_arr_DE, corr_DE, c='k', label='DE Mean', ls='--', lw=3)

plt.ylim(-0.2,1)
plt.xlim(0,8)
plt.xlabel('Lead Time [Months]')
plt.ylabel('r')
#plt.title('Correlation skill')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

#plt.savefig(join(plotdir, f'compare_corr_{start[2:4]}{end[2:4]}.pdf'))





ax = plt.figure(figsize=(6,3)).gca()

plt.plot(lead_arr_of, ssrmse_UBC_NNET, label='UBC NNET', ls='--')
plt.plot(lead_arr_of, ssrmse_UCLA_TCD, label='UCLA-TCD',  ls='--')
plt.plot(lead_arr_of, ssrmse_CPC_MRKOV, label='CPC MRKOV', ls='--')
plt.plot(lead_arr_of, ssrmse_CPC_CA, label='CPC CA', ls='--')
plt.plot(lead_arr_of, ssrmse_CPC_CCA, label='CPC CCA', ls='--')

plt.plot(lead_arr_of, ssrmse_NASA_GMAO, label='NASA GMAO')
plt.plot(lead_arr_of, ssrmse_NCEP_CFS, label='NCEP CFS')
plt.plot(lead_arr_of, ssrmse_JMA, label='JMA')
plt.plot(lead_arr_of, ssrmse_SCRIPPS, label='SCRIPPS')
plt.plot(lead_arr_of, ssrmse_ECMWF, label='ECMWF')
plt.plot(lead_arr_of, ssrmse_KMA_SNU, label='KMA SNU')

plt.plot(lead_arr_DE, ssrmse_DE, c='k', label='DE Mean', ls='--', lw=3)

plt.ylim(0., 1.8)
plt.xlim(0,8)
plt.xlabel('Lead Time [Months]')
plt.ylabel('SRMSE')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

#plt.savefig(join(plotdir, f'compare_ssrmse_{start[2:4]}{end[2:4]}.pdf'))
