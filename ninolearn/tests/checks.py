import inspect
import numpy as np
# =============================================================================
# Checks are tests that are repeated for multiple modules/classes
# =============================================================================

def check_core_methods(model):
    """
    Performs a test if a model contains all the core methods that a ML-model
    in NinoLearn should have.

    :param model: The model that is tested.
    """
    assert inspect.isfunction(model.set_parameters)
    assert inspect.isfunction(model.fit)
    assert inspect.isfunction(model.predict)
    assert inspect.isfunction(model.evaluate)
    assert inspect.isfunction(model.save)
    assert inspect.isfunction(model.load)

def check_regularizer(layer_config, class_name='L1L2', l1=None, l2=None):
    """
    Check if the kernel L1 and/or L2 regularizer coefficients are (about) the
    same as the one that one expects. The regularizer coefficients are in
    general not exactly the same as the chosen one (because of Keras), but it
    should be about the same. This is why np.round is used in this check.

    :type layer_config: dict
    :param layer_config: the layer configuraion dictionary.

    :type class_name: str
    :param class_name: The expected class name of the regularizer

    :type l1,l2: float
    :param l1,l2: the expected coefficent values of the l1 and/or coefficients.
    """

    kernel_regularizer = layer_config['kernel_regularizer']
    assert kernel_regularizer['class_name'] == class_name
    regularizer_config = kernel_regularizer['config']

    if l1 is not None:
        assert np.round(regularizer_config['l1']/l1, 1) == 1.
    if l2 is not None:
        assert np.round(regularizer_config['l2']/l2, 1) == 1.