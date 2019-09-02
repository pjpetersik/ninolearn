import numpy as np


import keras.backend as K
from keras.models import Model, save_model, load_model
from keras.layers import Dense, Input, Dropout
from keras.layers import GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers

from os.path import join, exists
from os import mkdir, listdir, getcwd
from shutil import rmtree


from ninolearn.learn.skillMeasures import rmse
from ninolearn.exceptions import MissingArgumentError
from ninolearn.utils import small_print_header, print_header


import warnings

class EncoderDecoder(object):
    """
    The Encoder-Decoder is an neural network that has the same architecture as
    an Autoencoder. Hence, labal and feature vector have the same dimension.
    In the ninolearn package the model is called Encoder-Decoder because it is
    used for prediction purposes and therefore label and feature vector might
    be soparated by some time lag or even are not the same variable.
    """
    def set_parameters(self, neurons=(128, 16), dropout=0.2, noise=0.2,
                       noise_out=0.2, l1_hidden=0.0001, l2_hidden=0.0001,
                       l1_out=0.0001, l2_out=0.0001, batch_size=50,
                       lr=0.0001, n_segments=5, n_members_segment=1,
                       patience = 40, epochs=500, verbose=0):
        """
        Set the parameters of the Encoder-Decoder neural network.

        Note, if the parameters are given in a list, ninolearn assumes that
        a the method .fit_RandomizedSearch() is used.

        :type neurons: tuple (list of two tuples for .fit_RandomizedSearch())
        :param neurons: The architecture of the Encoder-Decoder. The layer\
        with the lowest number of neurons is assumed to be the bottleneck layer\
        for which the activation function is linear. Furthermore, the output\
        layer has a linear activation as well. All other layers have the ReLU\
        as activation.

        :type dropout: float
        :param dropout: Standard deviation of the Gaussian dropout. Dropout\
        layers are installed behind each hidden layer in the Encoder and the\
        Decoder.

        :type noise: float
        :param noise: Standard deviation of Gaussian noise for the input layer.

        :type noise: float
        :param noise: Standard deviation of Gaussian noise for the output\
        layer.

        :type l1_hidden,  l2_hidden: float
        :param l1_hidden, l2_hidden: Coefficent for the L1 and the L2 penalty\
        term for the hidden layer weights.

        :type l1_out,  l2_out: float
        :param l1_hidden, l2_hidden: Coefficent for the L1 and the L2 penalty\
        term for the output layer weights.

        :type batch_size: int
        :param batch_size: Batch size  during training of a member of the\
        Encoder-Decoder.

        :type lr: float
        :param lr: The learning rate.

        :type n_segments: int
        :param n_segments: The number of segments that are used for the cross-\
        validation scheme and the training of the Ensemble members.

        :type n_members_segment:  int
        :param n_members_segment: The number of members that are trained for\
        one segment.

        :type patience: int
        :param patience: The number of epochs to wait until Early-Stopping\
        stops the training.

        :type epochs: int
        :param epochs: The maximum number of epochs.

        :type verbose: int
        :param verbose: Print some progress to screen. Either 0 (silent), 1 or\
        2.
        """

        # hyperparameters
        self.hyperparameters = {'neurons': neurons,
                'dropout': dropout, 'noise': noise, 'noise_out': noise_out,
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


    def build_model(self, n_features, n_labels):
        """
        The method builds a new member of the ensemble and returns it.

        :type n_features: int
        :param n_features: The number of features.

        :type n_labels: int
        :param n_labels: The number of labels.
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
            h = Dropout(self.hyperparameters['dropout'])(h)

        latent = Dense(self.hyperparameters['neurons'][self.bottelneck_layer], activation='linear',
                      kernel_regularizer=regularizers.l1_l2(self.hyperparameters['l1_hidden'],
                                                            self.hyperparameters['l2_hidden']))(h)

        encoder = Model(inputs=inputs, outputs=latent, name='encoder')

        # the decoder part
        latent_inputs = Input(shape=(self.hyperparameters['neurons'][self.bottelneck_layer],))
        h = GaussianNoise(0.0)(latent_inputs)

        for i in range(self.bottelneck_layer + 1, self.n_hidden_layers - 1):
            h = Dense(self.hyperparameters['neurons'][i], activation='relu',
                      kernel_regularizer=regularizers.l1_l2(self.hyperparameters['l1_hidden'],
                                                            self.hyperparameters['l2_hidden']))(h)
            h = Dropout(self.hyperparameters['dropout'])(h)

        decoded = Dense(n_labels, activation='linear',
                        kernel_regularizer=regularizers.l1_l2(self.hyperparameters['l1_out'],
                                                              self.hyperparameters['l2_out']))(h)

        decoder = Model(inputs=latent_inputs, outputs=decoded, name='decoder')

        # endocder-decoder model
        encoder_decoder = Model(inputs, decoder(encoder(inputs)), name='encoder_decoder')

        return encoder_decoder, encoder, decoder


    def fit(self, trainX, trainy, valX=None, valy=None):
        """
        Fit the model. If n_segments is 1, then a validation data set needs to
        be supplied.

        :type trainX: np.ndarray
        :param trainX: The training feature set. 2-D array with dimensions\
        (timesteps, features)

        :type trainy: np.ndarray
        :param trainy: The training label set. 2-D array with dimensions\
        (timesteps, labels)

        :type valX: np.ndarray
        :param valX: The validation feature set. 2-D array with dimensions\
        (timesteps, features).

        :type valy:  np.ndarray
        :param valy: The validation label set. 2-D array with dimensions\
        (timesteps, labels).
        """

        # clear memory
        K.clear_session()

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
                member, member_encoder, member_decoder = self.build_model(trainX.shape[1], trainy.shape[1])

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

    def fit_RandomizedSearch(self, trainX, trainy,  n_iter=10, **kwargs):
        """
        Hyperparameter optimazation using random search.


        :type trainX: np.ndarray
        :param trainX: The training feature set. 2-D array with dimensions\
        (timesteps, features).

        :type trainy: np.ndarray
        :param trainy: The training label set. 2-D array with dimensions\
        (timesteps, labels).

        :param kwargs: Keyword arguments are passed to the .fit() method.
        """
        # check if hyperparameters where provided in lists for randomized search
        if len(self.hyperparameters_search) == 0:
            raise Exception("No variable indicated for hyperparameter search!")

        #iterate with randomized hyperparameters
        best_loss = np.inf
        for i in range(n_iter):
            print_header(f"Search iteration Nr {i+1}/{n_iter}")

            # random selection of hyperparameters
            for key in self.hyperparameters_search.keys():
                low = self.hyperparameters_search[key][0]
                high = self.hyperparameters_search[key][1]

                if type(low) is float and type(high) is float:
                    self.hyperparameters[key] = np.random.uniform(low, high)

                if type(low) is int and type(high) is int:
                    self.hyperparameters[key] = np.random.randint(low, high+1)

                if type(low) is tuple and type(high) is tuple:
                    hyp_list = []
                    for i in range(len(low)):
                        hyp_list.append(np.random.randint(low[i], high[i]+1))
                    self.hyperparameters[key] = tuple(hyp_list)

            self.fit(trainX, trainy, **kwargs)

            # check if validation score was enhanced
            if self.mean_val_loss<best_loss:
                best_loss = self.mean_val_loss
                self.best_hyperparameters = self.hyperparameters.copy()

                small_print_header("New best hyperparameters")
                print(f"Mean loss: {best_loss}")
                print(self.best_hyperparameters)

        # refit the model with optimized hyperparameter
        # AND to have the weights of the DE for the best hyperparameters again
        print_header("Refit the model with best hyperparamters")

        self.hyperparameters = self.best_hyperparameters.copy()
        print(self.hyperparameters)
        self.fit(trainX, trainy, **kwargs)

        print(f"best loss search: {best_loss}")
        print(f"loss refitting : {self.mean_val_loss}")

    def predict(self, X):
        """
        Ensemble prediction.

        :type X: np.ndarray
        :param X: Feature set for which the prediction should be made.
        """
        pred_ens = np.zeros((X.shape[0], X.shape[1], self.n_members))
        for i in range(self.n_members):
            pred_ens[:,:,i] = self.ensemble[i].predict(X)
        return np.mean(pred_ens, axis=2), pred_ens

    def evaluate(self, X, ytrue):
        """
        Evaluate the model based on the RMSE

        :type X: np.ndarray
        :param X: The feature array.

        :type ytrue: np.ndarray
        :param ytrue: The true label array.
        """
        ypred, dummy = self.predict(X)
        return rmse(ytrue, ypred)


    def save(self, location='', dir_name='ed_ensemble'):
        """
        Save the ensemble.

        :type location: str
        :param location: Base directory where to for all Encoder-Decoder\
        ensembles

        :type dir_name: str
        :param dir_name: The specific directory name in the base directory\
        were to save the ensemble.
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
        Save the ensemble.

        :type location: str
        :param location: Base directory where for all Encoder-Decoder\
        ensembles.

        :type dir_name: str
        :param dir_name: The specific directory name in the base directory\
        were to find the ensemble.
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


