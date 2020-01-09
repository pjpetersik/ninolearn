# =============================================================================
# Here an example of an multilinear (Lasso) regression model is given that is
# based on the scikit-learn python package.
# =============================================================================

# =============================================================================
# First a data pipeline is build. The pipeline is used during training,
# prediction and evaluation to generate the feature, the label, the time
# as well as (optional) the persistance forecast.
# =============================================================================

# import the data reader to read data from the preprocessed data directory
from ninolearn.IO.read_processed import data_reader
import numpy as np

def pipeline(lead_time,  return_persistance=False):
    """
    Data pipeline for the processing of the data before the MLR
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
    # initialize the reader
    reader = data_reader(startdate='1960-01', enddate='2017-12')

    # load data
    oni = reader.read_csv('oni')
    wwv = reader.read_csv('wwv_proxy')
    iod = reader.read_csv('iod')

    # the shift data by 3 in addition to lead time shift (due to definition
    # of lead time) as in barnston et al. (2012)
    shift = 3
    # make feature
    Xorg = np.stack((oni, wwv, iod), axis=1)
    X = Xorg[:-lead_time-shift,:]


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


def pipeline_noise(lead_time,  return_persistance=False):
    """
    Data pipeline for the processing of the data before the MLR
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
    # initialize the reader
    reader = data_reader(startdate='1960-01', enddate='2017-12')

    np.random.seed(0)

    # load data
    oni = reader.read_csv('oni')
    wwv = reader.read_csv('wwv_proxy')
    iod = reader.read_csv('iod')

    # the shift data by 3 in addition to lead time shift (due to definition
    # of lead time) as in barnston et al. (2012)
    shift = 3

    # make feature
    Xorg = np.stack((oni, wwv, iod), axis=1)

    for i in range(100):
        random_noise = np.random.normal(size=len(oni)).reshape(len(oni), 1)
        Xorg = np.concatenate((Xorg, random_noise), axis=1)

    X = Xorg[:-lead_time-shift,:]


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


# =============================================================================
# Now the actual model is build.
# =============================================================================

# import the baseModel from which the mlr class needs to inherit
from ninolearn.learn.models.baseModel import baseModel

# import the sklearn model that we want to use for the ENSO forecast
from sklearn.linear_model import Lasso

# import some packages and methods to saving the model later
import pickle
from os.path import join, exists
from os import mkdir

# The actual model inherits from the baseModel class
class mlr(baseModel):

    # Two important variables of the class that each new model needs to define:
    # The number of outputs
    n_outputs=1
    # The name that is used when predictions are saved in an netCDF file.
    output_names = ['prediction']


    def __init__(self, alpha=1.0, name='mlr'):
        """
        The model needs to have an __init__ function. That takes contains
        receives the hyperparameters of the model as well as the name of the
        model as keyword arguments
        """
        # apply the .set_hyperparameters function, that was inherited from the
        # baseModel
        self.set_hyperparameters(alpha=alpha, name=name)

    def fit(self, trainX, trainy):
        """
        This is the fit function of the model. Very complex models, e.g. neural
        networks would need to split the trainX and trainy variables further
        to generate a validation data set, which is than used to calculate
        the self.mean_val_loss and to check for overfitting.
        Here, we don't need to do so because the model is not very complex and
        we have plenty of data to train the model.
        """
        #Initialize the Lasso model form the sci-kit learn package
        self.model = Lasso(self.hyperparameters['alpha'])

        # fit the model to the training data
        self.model.fit(trainX,trainy)

        # IMPORTANT: save the Score under self.mean_val_loss. This variable
        # will be used to be optimized during the random search later
        self.mean_val_loss = self.model.score(trainX, trainy)

    def predict(self, X):
        """
        Prediction function is rather simple when sci-kit learn models are used.
        """
        return self.model.predict(X)

    def save(self, location='', dir_name='mlr'):
        """
        Arguments of this function are mandetory and used to systemically
        save models in your modeldir.
        """
        path = join(location, dir_name)
        if not exists(path):
            mkdir(path)
        filename = join(path,f'model.sav')
        pickle.dump(self.model, open(filename, 'wb'))

    def load(self, location='', dir_name='mlr'):
        """
        Arguments of this function are mandetory and used to systemically
        load models from your modeldir.
        """
        path = join(location, dir_name)
        filename = join(path,f'model.sav')
        self.model = pickle.load(open(filename, 'rb'))

