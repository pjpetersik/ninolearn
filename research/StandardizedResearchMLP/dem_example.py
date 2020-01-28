import numpy as np
from sklearn.preprocessing import StandardScaler

from ninolearn.learn.models.dem import DEM


def pipeline_example():

    t = np.arange(10000)
    f1 = 0.01
    f2 = 0.001

    yraw = np.sin(f1*t*np.pi)
    z = np.sin(f2*t*np.pi) +1
    y = np.zeros_like(yraw)

    for i in range(len(y)):
       y[i] = yraw[i] + np.random.normal(scale=z[i])

    # process features
    feature_unscaled = np.stack((yraw, z), axis=1)

    # scale each feature
    scalerX = StandardScaler()
    X= scalerX.fit_transform(feature_unscaled)



    return X, y


if __name__=="__main__":
    X,y = pipeline_example()
    Xtrain, Xval, Xtest = test

    model = DEM(layers=1, neurons = 16, dropout=0.0, noise_in=0.0, noise_sigma=0.,
                 noise_mu=0., l1_hidden=0, l2_hidden=0.,
                   l1_mu=0., l2_mu=0., l1_sigma=0.0,
                   l2_sigma=0.0, lr=0.001, batch_size=100, epochs=5000, n_segments=5, n_members_segment=1, patience=100,
                   verbose=0, pdf="normal",  name="dem_example")


    model.fit()


