"""
here I want to implement the code for the deep ensemble
"""
import numpy as np

import keras.backend as K
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Input, concatenate
from keras.layers import Dropout, GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers

from os.path import join, exists
from os import mkdir, listdir, getcwd
from shutil import rmtree

from ninolearn.learn.losses import nll_gaussian
from ninolearn.learn.evaluation import rmse
from ninolearn.utils import print_header, small_print_header

import warnings
def _mixture(pred):
    """
    returns the ensemble mixture results
    """
    mix_mean = pred[:,0,:].mean(axis=1)
    mix_var = np.mean(pred[:,0,:]**2 + pred[:,1,:]**2, axis=1)  - mix_mean**2
    mix_std = np.sqrt(mix_var)
    return mix_mean, mix_std

def predict_ens(model_ens, X):
    """
    generates the ensemble prediction of a model ensemble

    :param model_ens: list of ensemble models
    :param X: the features
    """
    pred_ens = np.zeros((X.shape[0], 2, len(model_ens)))
    for i in range(len(model_ens)):
        pred_ens[:,:,i] = model_ens[i].predict(X)
    return _mixture(pred_ens)

def nll(mean_y, mean_pred, std_pred):
    """
    Negative - log -likelihood for the prediction of a gaussian probability
    """
    mean = mean_pred
    sigma = std_pred + 1e-6 # adding 1-e6 for numerical stability reasons

    first  =  0.5 * np.log(np.square(sigma))
    second =  np.square(mean - mean_y) / (2  * np.square(sigma))
    summed = first + second

    loss =  np.mean(summed, axis=-1)
    return loss

class DEM(object):
    def set_parameters(self, layers=1, neurons=16, dropout=0.2, noise=0.1,
                 l1_hidden=0.1, l2_hidden=0.1, l1_out=0.0, l2_out=0.1,
                 n_segments=5, n_members_segment=1, lr=0.001, patience = 10, epochs=300,
                 verbose=0, std=True):
        """
        A deep ensemble model (DEM) predicting mean (and standard deviation) with one hidden
        layer having the ReLU function as activation for the hidden layer. It
        is trained using the negative-log-likelihood of a gaussian distribution.
        :type int:
        :param neurons: Number of neurons.

        :type dropout: float
        :param dropout: Dropout rate.
        """
        self.layers = layers
        self.neurons = neurons
        self.dropout = dropout
        self.noise = noise
        self.l1_hidden = l1_hidden
        self.l2_hidden = l2_hidden
        self.l1_out = l1_out
        self.l2_out = l2_out
        self.std = std
        self.lr = lr
        self.patience = patience
        self.epochs = epochs
        self.verbose = verbose

        self.n_segments = n_segments
        self.n_members_segments = n_members_segment

        self.n_members = self.n_segments * self.n_members_segments
        self.std = std
        if self.std:
            self.loss = nll_gaussian
            self.loss_name = 'nll_gaussian'
        else:
            self.loss = 'mse'
            self.loss_name = 'mean_squared_error'


        self.optimizer = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
        self.es = EarlyStopping(monitor=f'val_{self.loss_name}', min_delta=0.0, patience=self.patience, verbose=0,
                   mode='min', restore_best_weights=True)

    def build_model(self, n_features):
        """
        The method builds a new member of the ensemble and returns it.
        """
        inputs = Input(shape=(n_features,))
        h = GaussianNoise(self.noise)(inputs)

        for _ in range(self.layers):
            h = Dense(self.neurons, activation='relu', kernel_regularizer=regularizers.l1_l2(self.l1_hidden, self.l2_hidden))(h)
            h = Dropout(self.dropout)(h)

        mu = Dense(1, activation='linear', kernel_regularizer=regularizers.l1_l2(self.l1_out, self.l2_out))(h)
        if self.std:
            sigma = Dense(1, activation='softplus', kernel_regularizer=regularizers.l1_l2(self.l1_out, self.l2_out))(h)
            outputs = concatenate([mu, sigma])

        else:
            outputs = mu

        model = Model(inputs=inputs, outputs=outputs)
        return model


    def fit(self, trainX, trainy, valX=None, valy=None):
        self.ensemble = []
        self.history = []

        self.segment_len = trainX.shape[0]//self.n_segments

        if self.n_segments==1 and (valX is not None or valy is not None):
             warnings.warn("Validation and test data set are the same if n_segements is 1!")

        i = 0
        while i<self.n_members_segments:
            j=0
            while j<self.n_segments:
                n_ens_sel = len(self.ensemble)
                small_print_header(f"Train member Nr {n_ens_sel+1}/{self.n_members}")

                ensemble_member = self.build_model(trainX.shape[1])

                ensemble_member.compile(loss=self.loss, optimizer=self.optimizer, metrics=[self.loss])

                # validate on the spare segment
                if self.n_segments!=1:
                    if valX is not None or valy is not None:
                        warnings.warn("Validation data set will be one of the segments. The provided validation data set is not used!")

                    start_ind = j * self.segment_len
                    end_ind = (j+1) *  self.segment_len

                    trainXens = np.delete(trainX, np.s_[start_ind:end_ind], axis=0)
                    trainyens = np.delete(trainy, np.s_[start_ind:end_ind])
                    valXens = trainX[start_ind:end_ind]
                    valyens = trainy[start_ind:end_ind]

                # validate on test data set
                elif self.n_segments==1:
                    if valX is None or valy is None:
                        raise ValueError("When segments length is 1, a validation data set must be provided")
                    trainXens = trainX
                    trainyens = trainy
                    valXens = valX
                    valyens = valy


                history = ensemble_member.fit(trainXens, trainyens,
                                            epochs=300, batch_size=1, verbose=self.verbose,
                                            shuffle=True, callbacks=[self.es],
                                            validation_data=(valXens, valyens))

                self.history.append(history)

                self.ensemble.append(ensemble_member)
                j+=1
            i+=1

    def predict(self, X, std=True):
        """
        Senerates the ensemble prediction of a model ensemble

        :param model_ens: list of ensemble models
        :param X: the features
        """
        if std:
            pred_ens = np.zeros((X.shape[0], 2, self.n_members))
        else:
            pred_ens = np.zeros((X.shape[0], 1, self.n_members))

        for i in range(self.n_members):
            pred_ens[:,:,i] = self.ensemble[i].predict(X)
        return self._mixture(pred_ens, std)


    def _mixture(self, pred, std):
        """
        returns the ensemble mixture results
        """
        mix_mean = pred[:,0,:].mean(axis=1)
        if std:
            mix_var = np.mean(pred[:,0,:]**2 + pred[:,1,:]**2, axis=1)  - mix_mean**2
            mix_std = np.sqrt(mix_var)

        else:
            mix_std = None

        return mix_mean, mix_std

    def evaluate(self, ytrue, mean_pred, std_pred=False):
        """
        Negative - log -likelihood for the prediction of a gaussian probability
        """
        if  std_pred is None:
            return rmse(ytrue, mean_pred)

        else:
            mean = mean_pred
            sigma = std_pred + 1e-6 # adding 1-e6 for numerical stability reasons

            first  =  0.5 * np.log(np.square(sigma))
            second =  np.square(mean - ytrue) / (2  * np.square(sigma))
            summed = first + second

            loss =  np.mean(summed, axis=-1)
            return loss

    def save(self, location='', dir_name='ensemble'):
        """
        Save the ensemble
        """
        path = join(location, dir_name)
        if not exists(path):
            mkdir(path)
        else:
            rmtree(path)
            mkdir(path)

        for i in range(self.n_members):
            path_h5 = join(path, f"member{i}.h5")
            save_model(self.ensemble[i], path_h5, include_optimizer=False)

    def load(self, location=None,  dir_name='ensemble'):
        """
        Load the ensemble
        """
        if location is None:
            location = getcwd()
        path = join(location, dir_name)
        files = listdir(path)
        self.n_members = len(files)
        self.ensemble = []

        for file in files:
            file_path = join(path, file)
            self.ensemble.append(load_model(file_path))
