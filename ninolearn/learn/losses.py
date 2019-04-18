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

def l_uniform(ytrue, pred):
    """
    TO BE DONE
    """
    mu = pred[:,0]
    w = pred[:,1]

    a = mu - w
    b = mu + w

    comparison1 = K.less_equal(a, ytrue[:,0])
    comparison2 = K.greater_equal(b, ytrue[:,0])
    comparison = K.stack([comparison1,comparison2], axis=0)

    loss =  K.switch(K.all(comparison, axis=0), 1/(2*w), a * 0.)

    return  -K.prod(loss)

