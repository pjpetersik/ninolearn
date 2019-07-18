import keras.backend as K

def nll_gaussian(y_true, y_pred):
    """
    Negative - log -likelihood for the prediction of a gaussian probability
    """
    mean = y_pred[:,0]
    sigma = y_pred[:,1] + 1e-6 # adding 1-e6 for numerical stability reasons

    first  =  0.5 * K.log(K.square(sigma))
    second =  K.square(mean - y_true[:,0]) / (2  * K.square(sigma))
    summed = first + second

    loss =  K.mean(summed, axis=-1)
    return loss