"""
here I want to implement the code for the deep ensemble
"""
import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Input, concatenate
from keras.layers import Dropout, GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers

from sklearn.preprocessing import StandardScaler

from ninolearn.learn.losses import nll_gaussian
from ninolearn.utils import print_header

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

def evaluate_ens(mean_y, mean_pred, std_pred):
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
    def __init__(self, neurons=16, dropout=0.2, noise=0.1,
                 l1_hidden=0.1, l2_hidden=0.1, l1_out=0.0, l2_out=0.1,
                 n_segments=5, n_members_segment=1, lr=0.001, patience = 10, epochs=300):
        """
        A deep ensemble model (DEM) predicting mean and standard deviation with one hidden
        layer having the ReLU function as activation for the hidden layer. It
        is trained using the negative-log-likelihood of a gaussian distribution.

        :type X: np.ndarray
        :param X: The feature vector of shape (samples, features). Does not need to be scaled. It will be scaled
        internally.

        :type X: np.ndarray
        :param y: The target vector of shape (samples,) .

        :type int:
        :param neurons: Number of neurons.

        :type dropout: float
        :param dropout: Dropout rate.
        """
        self.neurons = neurons
        self.dropout = dropout
        self.noise = noise
        self.l1_hidden = l1_hidden
        self.l2_hidden = l2_hidden
        self.l1_out = l1_out
        self.l2_out = l2_out
        self.lr = lr
        self.patience = patience
        self.epochs = epochs

        self.optimizer = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
        self.es = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=self.patience, verbose=0,
                   mode='min', restore_best_weights=True)

        self.n_segments = n_segments
        self.n_members_segments = n_members_segment


        self.n_members = self.n_segments * self.n_members_segments
        self.ensemble = []
        self.history = []

    def build_model(self, n_features):
        """
        The method builds a new member of the ensemble and returns it.
        """
        inputs = Input(shape=(n_features,))
        h = GaussianNoise(self.noise)(inputs)

        h = Dense(self.neurons, activation='relu', kernel_regularizer=regularizers.l1_l2(self.l1_hidden, self.l2_hidden))(h)
        h = Dropout(self.dropout)(h)

        mu = Dense(1, activation='linear', kernel_regularizer=regularizers.l1_l2(self.l1_out, self.l2_out))(h)
        sigma = Dense(1, activation='softplus', kernel_regularizer=regularizers.l1_l2(self.l1_out, self.l2_out))(h)

        outputs = concatenate([mu, sigma])
        model = Model(inputs=inputs, outputs=outputs)
        return model


    def fit(self, trainX, trainy, valX=None, valy=None):
        self.segment_len = trainX.shape[0]//self.n_segments

        i = 0
        while i<self.n_members_segments:
            j=0
            while j<self.n_segments:
                n_ens_sel = len(self.ensemble)
                print_header(f"Train iteration Nr {n_ens_sel}")

                ensemble_member = self.build_model(trainX.shape[1])
                ensemble_member.compile(loss=nll_gaussian, optimizer=self.optimizer, metrics=[nll_gaussian])

                # validate on the spare segment
                if self.n_segments!=1:
                    if valX is not None or valy is not None:
                        raise Warning("Validation data set will be one of the segments. The provided validation data set is not used!")

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
                    print("Validation and test data set are the same!")

                history = ensemble_member.fit(trainXens, trainyens,
                                            epochs=300, batch_size=1, verbose=1,
                                            shuffle=True, callbacks=[self.es],
                                            validation_data=(valXens, valyens))

                self.history.append(history)

                self.ensemble.append(ensemble_member)
                j+=1
            i+=1

    def predict(self, X):
        """
        generates the ensemble prediction of a model ensemble

        :param model_ens: list of ensemble models
        :param X: the features
        """
        pred_ens = np.zeros((X.shape[0], 2, self.n_members))

        for i in range(self.n_members):
            pred_ens[:,:,i] = self.ensemble[i].predict(X)
        return self._mixture(pred_ens)


    def _mixture(self, pred):
        """
        returns the ensemble mixture results
        """
        mix_mean = pred[:,0,:].mean(axis=1)
        mix_var = np.mean(pred[:,0,:]**2 + pred[:,1,:]**2, axis=1)  - mix_mean**2
        mix_std = np.sqrt(mix_var)
        return mix_mean, mix_std

    def evaluate(self, ytrue, mean_pred, std_pred):
        """
        Negative - log -likelihood for the prediction of a gaussian probability
        """
        mean = mean_pred
        sigma = std_pred + 1e-6 # adding 1-e6 for numerical stability reasons

        first  =  0.5 * np.log(np.square(sigma))
        second =  np.square(mean - ytrue) / (2  * np.square(sigma))
        summed = first + second

        loss =  np.mean(summed, axis=-1)
        return loss



