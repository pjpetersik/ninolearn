from os.path import join
import xarray as xr
from ninolearn.private import plotdir
from ninolearn.pathes import processeddir
import matplotlib.pyplot as plt
from ninolearn.IO.read_processed import data_reader
from ninolearn.plot.prediction import plot_prediction


plt.close('all')

start = '2003-01'
end = '2017-12'
reader = data_reader(startdate=start, enddate=end)
oni = reader.read_csv('oni')

data = xr.open_dataset(join(processeddir, f'dem_forecasts.nc'))


fig, axs = plt.subplots(3, figsize=(9,7))

lead_arr = [0, 3, 6]


for i in range(len(lead_arr)):
    lead = lead_arr[i]
    lead_DE = lead//3

    UU_DE_mean = data['mean'].loc[start:end][:,lead_DE]
    UU_DE_std = data['std'].loc[start:end][:,lead_DE]

    plot_prediction(UU_DE_mean.target_season.values, UU_DE_mean.values,std=UU_DE_std.values,
                    alpha=0.4, ax=axs[i])

    NASA_GMAO = reader.read_other_forecasts('NASA GMAO', lead)
    NCEP_CFS = reader.read_other_forecasts('NCEP CFS', lead)
    JMA = reader.read_other_forecasts('JMA', lead)
    SCRIPPS =  reader.read_other_forecasts('SCRIPPS', lead)
    ECMWF = reader.read_other_forecasts('ECMWF', lead)
    KMA_SNU = reader.read_other_forecasts('KMA SNU', lead)
    UBC_NNET = reader.read_other_forecasts('UBC NNET', lead)
    UCLA_TCD = reader.read_other_forecasts('UCLA-TCD', lead)
    CPC_MRKOV = reader.read_other_forecasts('CPC MRKOV', lead)
    CPC_CA = reader.read_other_forecasts('CPC CA', lead)
    CPC_CCA = reader.read_other_forecasts('CPC CCA', lead)

    target_season = NASA_GMAO.target_season

    #%%
    alpha = 1
    axs[i].plot(target_season, UBC_NNET, alpha=alpha, label='UBC NNET', ls='--')
    axs[i].plot(target_season, UCLA_TCD, alpha=alpha, label='UCLA-TCD', ls='--')
    axs[i].plot(target_season, CPC_MRKOV, alpha=alpha, label='CPC MRKOV', ls='--')
    axs[i].plot(target_season, CPC_CA, alpha=alpha, label='CPC CA', ls='--')
    axs[i].plot(target_season, CPC_CA, alpha=alpha, label='CPC CCA', ls='--')

    axs[i].plot(target_season, NASA_GMAO, alpha=alpha, label='NASA GMAO')
    axs[i].plot(target_season, NCEP_CFS, alpha=alpha, label='NCEP CFS')
    axs[i].plot(target_season, JMA, alpha=alpha, label='JMA')
    axs[i].plot(target_season, SCRIPPS, alpha=alpha, label='SCRIPPS')
    axs[i].plot(target_season, ECMWF, alpha=alpha, label='ECMWF')
    axs[i].plot(target_season, KMA_SNU, alpha=alpha, label='KMA SNU')

    axs[i].plot(oni.index, oni, c='r', label='ONI', lw=2)

    axs[i].set_xlim(UU_DE_mean.target_season.values[0], UU_DE_mean.target_season.values[-1])
    axs[i].set_ylim(-3,3)

    if i==1:
        axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if i==2:
        axs[i].set_xlabel('Time [Year]')

    axs[i].axhspan(-0.5, -6, facecolor='blue',  alpha=0.1,zorder=0)
    axs[i].axhspan(0.5, 6, facecolor='red', alpha=0.1,zorder=0)


    #axs[i].set_title(f"Lead time: {lead} months")
    axs[i].grid()

    axs[i].set_ylabel('ONI [K]')

    axs[i].text(oni.index[4], 2.2, f'{lead}-months', weight='bold', size=10,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 7})
    #plt.tight_layout()
    plt.subplots_adjust(top=0.945,
                        bottom=0.095,
                        left=0.09,
                        right=0.795,
                        hspace=0.2,
                        wspace=0.2)

    #plt.savefig(join(plotdir, f'compare_prediction_lead{lead}.pdf'))

