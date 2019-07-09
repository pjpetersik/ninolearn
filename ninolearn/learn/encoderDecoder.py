import numpy as np


import keras.backend as K
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Input
from keras.layers import GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers

from os.path import join, exists
from os import mkdir, listdir, getcwd
from shutil import rmtree

from ninolearn.exceptions import MissingArgumentError
from ninolearn.utils import small_print_header

import warnings

class EncoderDecoder(object):
    def set_parameters(self, neurons=[128, 16], dropout=0.2, noise=0.2, noise_out=0.0,
                 l1_hidden=0.0001, l2_hidden=0.0001, l1_out=0.0001, l2_out=0.0001, batch_size=50,
                 lr=0.0001, n_segments=5, n_members_segment=1, patience = 40, epochs=500, verbose=0):

        """
        Set the parameters of the Encoder-Decoder neural network
        """

        # hyperparameters
        self.hyperparameters = {'neurons': neurons,
                'dropout': dropout, 'noise': noise, 'noise_out':noise,
                'l1_hidden': l1_hidden, 'l2_hidden': l2_hidden,
                'l1_out': l1_out, 'l2_out': l2_out,
                'lr': lr, 'batch_size': batch_size}

        self.n_hidden_layers = len(neurons)
        self.bottelneck_layer = np.argmin(neurons)

        # hyperparameters for randomized search
        self.hyperparameters_search = {}

        for key in self.hyperparameters.keys():
            if type(self.hyperparameters[key]) is list:
                if len(self.hyperparameters[key])>0:
                    self.hyperparameters_search[key] = self.hyperparameters[key].copy()

        # training settings
        self.patience = patience
        self.epochs = epochs
        self.verbose = verbose

        # traning/validation split
        self.n_segments = n_segments
        self.n_members_segment = n_members_segment

        # derived parameters
        self.n_members = self.n_segments * self.n_members_segment


    def build_model(self, n_features):
        """
        The method builds a new member of the ensemble and returns it.
        """
        # initialize optimizer and early stopping
        self.optimizer = Adam(lr=self.hyperparameters['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False)
        self.es = EarlyStopping(monitor=f'val_mean_squared_error', min_delta=0.0, patience=self.patience, verbose=1,
                   mode='min', restore_best_weights=True)


        inputs = Input(shape=(n_features,))
        h = GaussianNoise(self.hyperparameters['noise'])(inputs)

        # the encoder part
        for i in range(self.bottelneck_layer):
            h = Dense(self.hyperparameters['neurons'][i], activation='relu',
                      kernel_regularizer=regularizers.l1_l2(self.hyperparameters['l1_hidden'],
                                                            self.hyperparameters['l2_hidden']))(h)

        latent = Dense(self.hyperparameters['neurons'][self.bottelneck_layer], activation='linear',
                      kernel_regularizer=regularizers.l1_l2(self.hyperparameters['l1_hidden'],
                                                            self.hyperparameters['l2_hidden']))(h)

        self.encoder = Model(inputs=inputs, outputs=latent, name='encoder')

        # the decoder part
        latent_inputs = Input(shape=(self.hyperparameters['neurons'][self.bottelneck_layer],))
        h = GaussianNoise(0.0)(latent_inputs)

        for i in range(self.bottelneck_layer + 1, self.n_hidden_layers - 1):
            h = Dense(self.hyperparameters['neurons'][i], activation='relu',
                      kernel_regularizer=regularizers.l1_l2(self.hyperparameters['l1_hidden'],
                                                            self.hyperparameters['l2_hidden']))(h)

        decoded = Dense(n_features, activation='linear',
                        kernel_regularizer=regularizers.l1_l2(self.hyperparameters['l1_out'],
                                                              self.hyperparameters['l2_out']))(h)

        self.decoder = Model(inputs=latent_inputs, outputs=decoded, name='decoder')

        # endocder-decoder model
        encoder_decoder = Model(inputs, self.decoder(self.encoder(inputs)), name='encoder_decoder')

        return encoder_decoder


    def fit(self, trainX, trainy,valX=None, valy=None):
        # clear memory
        K.clear_session

         # allocate lists for the ensemble
        self.ensemble = []
        self.history = []
        self.val_loss = []

        self.segment_len = trainX.shape[0]//self.n_segments

        if self.n_segments==1 and (valX is not None or valy is not None):
             warnings.warn("Validation and test data set are the same if n_segements is 1!")

        i = 0
        while i<self.n_members_segment:
            j = 0
            while j<self.n_segments:
                n_ens_sel = len(self.ensemble)
                small_print_header(f"Train member Nr {n_ens_sel+1}/{self.n_members}")

                # build model
                member = self.build_model(trainX.shape[1])

                # compite model
                member.compile(loss='mse', optimizer=self.optimizer, metrics=['mse'])

                # validate on the spare segment
                if self.n_segments!=1:
                    if valX is not None or valy is not None:
                        warnings.warn("Validation data set will be one of the segments. The provided validation data set is not used!")

                    start_ind = j * self.segment_len
                    end_ind = (j+1) *  self.segment_len

                    trainXens = np.delete(trainX, np.s_[start_ind:end_ind], axis=0)
                    trainyens = np.delete(trainy, np.s_[start_ind:end_ind], axis=0)
                    valXens = trainX[start_ind:end_ind]
                    valyens = trainy[start_ind:end_ind]
                    print

                # validate on test data set
                elif self.n_segments==1:
                    if valX is None or valy is None:
                        raise MissingArgumentError("When segments length is 1, a validation data set must be provided.")
                    trainXens = trainX
                    trainyens = trainy
                    valXens = valX
                    valyens = valy

                history = member.fit(trainXens, trainyens,
                                            epochs=self.epochs, batch_size=self.hyperparameters['batch_size'],
                                            verbose=self.verbose,
                                            shuffle=True, callbacks=[self.es],
                                            validation_data=(valXens, valyens))

                self.history.append(history)
                self.val_loss.append(member.evaluate(valXens, valyens)[1])
                print(f"Loss: {self.val_loss[-1]}")
                self.ensemble.append(member)
                j+=1
            i+=1
        self.mean_val_loss = np.mean(self.val_loss)
        print(f"Mean loss: {self.mean_val_loss}")

    def predict(self, X):
        """
        Ensemble prediction
        """
        pred_ens = np.zeros((X.shape[0], X.shape[1], self.n_members))
        for i in range(self.n_members):
            pred_ens[:,:,i] = self.ensemble[i].predict(X)
        return np.mean(pred_ens, axis=2), pred_ens


    def save(self, location='', dir_name='ed_ensemble'):
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

        output_neurons = self.ensemble[0].get_output_shape_at(0)[1]
        if output_neurons==2:
            self.std = True
        else:
            self.std = False


