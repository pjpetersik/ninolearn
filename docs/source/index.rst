.. ninolearn documentation master file, created by
   sphinx-quickstart on Thu Jul 18 11:31:57 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NinoLearn's documentation!
=====================================

Documentation for the Code
##########################
.. toctree::
   :maxdepth: 2
   :caption: Contents:

Ninolearn is a research framework for statistical ENSO prediction.

NinoLearn Download
##################
.. automodule:: ninolearn.download
    :members: download, downloadFTP, downloadHTTP, sources



NinoLearn Learn
###############
Models
******

Deep Ensemble Model (DEM)
-------------------------
.. automodule:: ninolearn.learn.models.dem
    :members:

Encoder-Decoder (ED)
--------------------
.. automodule:: ninolearn.learn.models.encoderDecoder
    :members:

NinoLearn Postprocess
#####################

Data preparation
****************
.. automodule:: ninolearn.postprocess.prepare
    :members:
.. automodule:: ninolearn.postprocess.anomaly
    :members:
.. automodule:: ninolearn.postprocess.regrid
    :members:

Evolving complex networks
*************************
.. automodule:: ninolearn.postprocess.network
    :members:

Principal Component Analysis
****************************
.. autoclass:: ninolearn.postprocess.pca.pca
    :members:

NinoLearn Input/Output
######################
Reading the raw data
********************
.. automodule:: ninolearn.IO.read_raw
   :members:

Reading the postprocessed data
******************************
.. autoclass:: ninolearn.IO.read_post.data_reader
   :members:

