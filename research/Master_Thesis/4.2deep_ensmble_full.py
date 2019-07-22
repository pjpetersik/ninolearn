import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K

from ninolearn.learn.models.dem import DEM
from ninolearn.utils import print_header, small_print_header
from ninolearn.pathes import modeldir

from data_pipeline import pipeline

import os
import time
plt.close("all")
K.clear_session()

#%% =============================================================================
# Deep ensemble
# =============================================================================
decades = [60, 70, 80, 90, 100, 110]
decades = [80]
for lead_time in [6, 0]:
    X, y, timey, yp = pipeline(lead_time, return_persistance=True)
    print_header(f'Lead time: {lead_time} month')
    for decade in decades:
        K.clear_session()
        small_print_header(f'Test period: {1902+decade}-01-01 till {1911+decade}-12-01')

        # jump loop iteration if already trained
        ens_dir=f'ensemble_decade{decade}_lead{lead_time}'
        out_dir = os.path.join(modeldir, ens_dir)

        modified_time = time.gmtime(os.path.getmtime(out_dir))
        compare_time = time.strptime("21-7-2019 13:00 UTC", "%d-%m-%Y %H:%M %Z")

        if modified_time>compare_time:
            print("Trained already!")
            continue

        test_indeces = (timey>=f'{1902+decade}-01-01') & (timey<=f'{1911+decade}-12-01')
        train_indeces = np.invert(test_indeces)

        trainX, trainy = X[train_indeces,:], y[train_indeces]

        model = DEM()

        model.set_parameters(layers=1, dropout=[0.1, 0.5], noise=[0.1,0.5], l1_hidden=[0.0, 0.2],
                    l2_hidden=[0, 0.2], l1_mu=[0.0, 0.2], l2_mu=[0.0, 0.2], l1_sigma=[0.0, 0.2],
                    l2_sigma=[0.0, 0.2],
                    lr=[0.0001,0.01], batch_size=100, epochs=500, n_segments=5,
                    n_members_segment=1, patience=30, verbose=0, std=True)

        model.fit_RandomizedSearch(trainX, trainy, n_iter=200)

        model.save(location=modeldir, dir_name=ens_dir)

        del model