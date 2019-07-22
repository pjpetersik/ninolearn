from ninolearn.learn.models.dem import DEM
from ninolearn.learn.models.encoderDecoder import EncoderDecoder

import inspect



def test_dem():
    check_core_methods(DEM)

def test_EncoderDecoder():
    check_core_methods(EncoderDecoder)

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


