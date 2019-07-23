from ninolearn.learn.models.dem import DEM
from ninolearn.learn.models.encoderDecoder import EncoderDecoder
from ninolearn.tests.checks import check_core_methods, check_regularizer
import numpy as np
import pytest

# =============================================================================
# Deep Ensemble
# =============================================================================
def test_DEM_Methods():
    check_core_methods(DEM)

def test_DEM_set_paramters():
    """
    This test checks if the model that one intends to build with the method
    .set_paramters() of an DEM instance is the one that is actually produced
    in the backend by Keras.
    """
    model = DEM()

    layers = np.random.randint(1, 5)
    neurons = np.random.randint(10, 500)
    n_features = np.random.randint(10, 500)

    l1_hidden = np.random.uniform(0,1)
    l2_hidden = np.random.uniform(0,1)
    l1_mu = np.random.uniform(0,1)
    l2_mu = np.random.uniform(0,1)
    l1_sigma = np.random.uniform(0,1)
    l2_sigma = np.random.uniform(0,1)

    dropout_rate = np.random.uniform(0,1)

    noise_in = np.random.uniform(0,1)
    noise_mu = np.random.uniform(0,1)
    noise_sigma = np.random.uniform(0,1)
    model.set_parameters(layers=layers, neurons=neurons, dropout=dropout_rate,
                 noise_in=noise_in, noise_sigma=noise_sigma, noise_mu=noise_mu,
                 l1_hidden=l1_hidden, l2_hidden=l2_hidden, l1_mu=l1_mu, l2_mu=l2_mu, l1_sigma=l1_sigma,
                 l2_sigma=l2_sigma, batch_size=10, n_segments=5, n_members_segment=1,
                 lr=0.001, patience = 10, epochs=300, verbose=0, std=True)


    member = model.build_model(n_features)

    # check input shape
    assert n_features == member.input_shape[1]

    # check input noise layer
    noise_in_config = member.get_layer(name=f'noise_input').get_config()
    assert noise_in_config['stddev'] == noise_in

    for i in range(model.hyperparameters['layers']):

        # check the hidden layer
        hidden_config = member.get_layer(name=f'hidden_{i}').get_config()
        assert hidden_config['activation'] == 'relu'
        assert hidden_config['units'] == neurons
        check_regularizer(hidden_config, class_name='L1L2', l1=l1_hidden, l2=l2_hidden)

        # check the dropout layer
        hidden_dropout_config = member.get_layer(name=f'hidden_dropout_{i}').get_config()
        assert hidden_dropout_config['rate'] == dropout_rate

    # check the mean output neuron
    mu = member.get_layer(name='mu_output')
    mu_config = mu.get_config()
    assert mu_config['activation'] == 'linear'
    check_regularizer(mu_config, class_name='L1L2', l1=l1_mu, l2=l2_mu)

    # check standard deviation output neuron
    sigma = member.get_layer(name='sigma_output')
    sigma_config = sigma.get_config()
    assert sigma_config['activation'] == 'softplus'
    check_regularizer(sigma_config, class_name='L1L2', l1=l1_sigma, l2=l2_sigma)

    # check mu noise layer
    noise_in_config = member.get_layer(name=f'noise_mu').get_config()
    assert noise_in_config['stddev'] == noise_mu

    # check sigma noise layer
    noise_in_config = member.get_layer(name=f'noise_sigma').get_config()
    assert noise_in_config['stddev'] == noise_sigma


def test_DEM_fit():
    pass
# =============================================================================
# Encoder Decoder model
# =============================================================================

def test_EncoderDecoder_Methods():
    check_core_methods(EncoderDecoder)

