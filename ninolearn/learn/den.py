"""
here I want to implement the code for the deep ensemble
"""
import numpy as np

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
