from ninolearn.learn.models.dem import DEM
from ninolearn.learn.fit import cross_hindcast, evaluation_correlation
from cross_training import pipeline

cross_hindcast(DEM, pipeline, 'dem')
r, p  = evaluation_correlation(DEM, 'dem')