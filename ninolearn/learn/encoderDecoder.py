import numpy as np


import keras.backend as K
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Input, concatenate
from keras.layers import Dropout, GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers

class EncoderDecoder(object):
    def set_parameters(self, neurons=[128, 16], dropout=0.2, noise=0.2, noise_out=0.0,
                 l1_hidden=0.0001, l2_hidden=0.0001, l1_out=0.0001, l2_out=0.0001, batch_size=50,
                 lr=0.0001, patience = 40, epochs=500, verbose=1):
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


    def fit(self, trainX, trainy, valX=None, valy=None):
        # clear memory
        K.clear_session

        self.encoder_decoder = self.build_model(trainX.shape[1])
        self.encoder_decoder.compile(loss='mse', optimizer=self.optimizer, metrics=['mse'])


        self.history = self.encoder_decoder.fit(trainX, trainy,
                  epochs=self.epochs, batch_size=self.hyperparameters['batch_size'],
                  verbose=self.verbose,
                  shuffle=True, callbacks=[self.es],
                  validation_data=(valX, valy))

    def predict(self, X):
        return self.encoder_decoder.predict(X)

