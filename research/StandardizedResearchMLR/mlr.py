"""
Here an example of an multilinear regression model is given.
"""


from ninolearn.learn.models.baseModel import baseModel
from ninolearn.IO.read_processed import data_reader

from sklearn.linear_model import Lasso
import numpy as np

def pipeline(lead_time,  return_persistance=False):
    """
    Data pipeline for the processing of the data before the Deep Ensemble
    is trained.

    :type lead_time: int
    :param lead_time: The lead time in month.

    :type return_persistance: boolean
    :param return_persistance: Return as the persistance as well.

    :returns: The feature "X" (at observation time), the label "y" (at lead
    time), the target season "timey" (least month) and if selected the
    label at observation time "y_persistance". Hence, the output comes as:
    X, y, timey, y_persistance.
    """
    shift = 3

    reader = data_reader(startdate='1960-01', enddate='2017-12')

    # indeces
    oni = reader.read_csv('oni')
    wwv = reader.read_csv('wwv_proxy')
    X = wwv[:-lead_time-shift,:]

    # arange label
    yorg = oni.values
    y = yorg[lead_time + shift:]

    # get the time axis of the label
    timey = oni.index[lead_time + shift:]

    if return_persistance:
        y_persistance = yorg[: - lead_time - shift]
        return X, y, timey, y_persistance
    else:
        return X, y, timey

class mlr(baseModel):
    def __init__(self,alpha=1.0):
        self.set_hyperparameters(alpha=alpha)

    def fit(self,X,y):
        self.model = Lasso(self.hyperparameters['alpha'])
        self.model.fit(X,y)

    def predict(self,X):
        self.model.predict(X)


