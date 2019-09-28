import numpy as np
from sklearn.preprocessing import StandardScaler

from ninolearn.utils import include_time_lag
from ninolearn.IO.read_processed import data_reader
from ninolearn.learn.models.dem import DEM

from ninolearn.learn.fit import cross_training


if __name__=="__main__":
    cross_training(DEM, pipeline, 50,
                   layers=1, dropout=[0.1, 0.5], noise_in=[0.1,0.5], noise_sigma=[0.1,0.5],
                   noise_mu=[0.1,0.5], l1_hidden=[0.0, 0.2], l2_hidden=[0, 0.2],
                   l1_mu=[0.0, 0.2], l2_mu=[0.0, 0.2], l1_sigma=[0.0, 0.2],
                   l2_sigma=[0.0, 0.2], lr=[0.0001,0.01], batch_size=100,
                   epochs=500, n_segments=5, n_members_segment=1, patience=30,
                   verbose=0, pdf="normal", name="dem")



