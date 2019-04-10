import keras.backend as K

def nll_gaussian(y_true, y_pred):
    """
    Negative - log -likelihood for the prediction of a gaussian probability
    """
    mean = y_pred[:,0]
    sigma = y_pred[:,1]

    first  =  0.5 * K.log(K.square(sigma))
    second =  K.square(mean - y_true[:,0]) / (2  * K.square(sigma))
    summed = first + second

    nll =  K.mean(summed, axis=-1)
    return nll


def nll_skew_gaussian(ytrue, pred):
    """
    TO BE DONE
    """
    pass