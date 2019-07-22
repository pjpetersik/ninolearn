from ninolearn import DEM
import inspect

def test_model():
    assert inspect.isfunction(DEM.save)
    assert inspect.isfunction(DEM.fit)
    assert inspect.isfunction(DEM.predict)
    assert inspect.isfunction(DEM.evaluate)
    assert inspect.isfunction(DEM.set_parameters)

