import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


from ninolearn.learn.fit import cross_hindcast, n_decades, lead_times, decade_color, decade_name
from ninolearn.learn.evaluation import evaluation_correlation, evaluation_decadal_correlation, evaluation_seasonal_correlation
from ninolearn.learn.evaluation import evaluation_srmse, evaluation_decadal_srmse, evaluation_seasonal_srmse
from ninolearn.plot.evaluation import plot_seasonal_skill

from mlr import pipeline, mlr, pipeline_noise
#

model_name = 'mlr_review'
cross_hindcast(mlr, pipeline, f'{model_name}')

#%% =============================================================================
# All season correlation skill
# =============================================================================

plt.close("all")
# scores on the full time series
r, p  = evaluation_correlation(f'{model_name}', variable_name='prediction')

# score in different decades
r_dec, p_dec = evaluation_decadal_correlation(f'{model_name}', variable_name='prediction')

# plot correlation skills
ax = plt.figure(figsize=(6.5,3.5)).gca()

for j in range(n_decades-1):
    plt.plot(lead_times, r_dec[:,j], c=decade_color[j], label=f"Deep Ens.  ({decade_name[j]})")
plt.plot(lead_times, r, label="Deep Ens.  (1962-2017)", c='k', lw=2)

plt.ylim(0,1)
plt.xlim(0.,lead_times[-1])
plt.xlabel('Lead Time [Months]')
plt.ylabel('r')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

#%% =============================================================================
# All season SRMSE skill
# =============================================================================
srmse_dec = evaluation_decadal_srmse(f'{model_name}', variable_name='prediction')
srmse = evaluation_srmse(f'{model_name}', variable_name='prediction')

# plot SRMSE skills
ax = plt.figure(figsize=(6.5,3.5)).gca()
for j in range(n_decades-1):
    plt.plot(lead_times, srmse_dec[:,j], c=decade_color[j], label=f"Deep Ens.  ({decade_name[j]})")
plt.plot(lead_times, srmse, label="Deep Ens.  (1962-2017)", c='k', lw=2)

plt.ylim(0,1.5)
plt.xlim(0.,lead_times[-1])
plt.xlabel('Lead Time [Months]')
plt.ylabel('SRMSE')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

# =============================================================================
# Seasonal correlation skills
# =============================================================================
# evaluate the model in different seasons
r_seas, p_seas = evaluation_seasonal_correlation(f'{model_name}', variable_name='prediction')

plot_seasonal_skill(lead_times, r_seas,  vmin=0, vmax=1)
plt.contour(np.arange(1,13),lead_times, p_seas, levels=[0.01, 0.05, 0.1], linestyles=['solid', 'dashed', 'dotted'], colors='k')
plt.title('Correlation skill')
plt.tight_layout()

srsme_seas = evaluation_seasonal_srmse(f'{model_name}', variable_name='prediction')
plot_seasonal_skill(lead_times, srsme_seas,  vmin=0, vmax=1.5, cmap=plt.cm.inferno_r, extend='max')
plt.title('SSRMSE skill')
plt.tight_layout()
