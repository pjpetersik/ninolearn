import matplotlib.pyplot as plt

#from ninolearn.learn.models.dem import DEM
from test import DEM
from ninolearn.learn.fit_predict import cross_fit, cross_predict
from ninolearn.learn.evaluation import evaluation_correlation, evaluation_decadal_correlation, evaluation_seasonal_correlation
from ninolearn.learn.evaluation import evaluation_srmse, evaluation_decadal_srmse, evaluation_seasonal_srmse

from pipeline import pipeline

# training
cross_fit(DEM, pipeline, 1, neurons=16,
               layers=1, dropout=[0.1, 0.5], noise_in=[0.1,0.5], noise_sigma=[0.1,0.5],
               noise_mu=[0.1,0.5], l1_hidden=[0.0, 0.2], l2_hidden=[0, 0.2],
               l1_mu=[0.0, 0.2], l2_mu=[0.0, 0.2], l1_sigma=[0.0, 0.2],
               l2_sigma=[0.0, 0.2], lr=[0.0001,0.01], batch_size=100,
               epochs=500, n_segments=5,
               n_members_segment=1, patience=30, verbose=0,
               pdf="normal", name="dem")

# make the hindcast that is used for the evaluation
cross_predict(DEM, pipeline, 'dem')

# evaluate the model onto the full time series
r, p  = evaluation_correlation('dem')
srmse = evaluation_srmse('dem')

# evaluate the model in different decades
r_dec, p_dec = evaluation_decadal_correlation('dem')
srmse_dec = evaluation_decadal_srmse('dem')

# evaluate the model in different seasons
r_seas, p_seas = evaluation_seasonal_correlation('dem')
srsme_seas = evaluation_seasonal_srmse('dem')
