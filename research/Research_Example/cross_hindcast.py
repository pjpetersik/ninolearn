from ninolearn.learn.models.dem import DEM
from ninolearn.learn.fit import cross_hindcast
from cross_training import pipeline

cross_hindcast(DEM, pipeline, 'dem_normal')
