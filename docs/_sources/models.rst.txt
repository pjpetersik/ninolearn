######
Models
######

The basic idea behind all of the NinoLearn models is that they inherit some
more general functions from the :class:`ninolearn.learn.models.baseModel.baseModel`
class. In this is done, such that all models can be trained following the same
standards. In particular, this is done such that the methods
:meth:`ninolearn.learn.fit.cross_training` and
:meth:`ninolearn.learn.fit.cross_hindcast` work equally for each new model.

**********
Base Model
**********
.. automodule:: ninolearn.learn.models.baseModel
    :members:

****************
Developed models
****************

Deep Ensemble Model (DEM)
=========================
.. automodule:: ninolearn.learn.models.dem
    :members:


Encoder-Decoder (ED)
====================
.. automodule:: ninolearn.learn.models.encoderDecoder
    :members: