############
Introduction
############

********************************
The El Niño Southern Oscillation
********************************

The El Niño Southern Oscillation (ENSO) is a coupelled ocean atmosphere phenomenon
which is present in the equatorial Pacific and affects the weather around the
world. In its positive phase, temperatures throught the equatorial Pacific are
relatively warm (El Niño phase). The other way around, temperatures are realtively
low in the negative phase (La Niña).

**************
ENSO forecasts
**************

The predictive horizon for ENSO forecasts is by far longer than
for weather forecasts, because of the strong autocorrelation of the ENSO for
time periods up to about 6-9 month. It is the combination of the relatively
long predictive horizon and the influence of ENSO on the weather around the
world, that raises the a great interest in research to make skillful seasonal
forecasts for the ENSO. Whereas, dynamical
models integrate physical equations that determine the evolution of the system
in time, statistical models *learn* from past observations how the future state
will likely evolve. You can find current forecasts from dynamical
and statistical ENSO-models on the website of the Internation Research Insitute for
Climate and Society (see `here <https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/>`_).

***************************
Existing statistical models
***************************

Multiple statistical models for the ENSO predictions have been
developed in past research. For instance a working group around
`William Hsieh <https://www.eoas.ubc.ca/people/williamhsieh>`_ at the
University of British Columbia (UBC) in Canada investigated the application of artifical
neural networks (ANN) for the ENSO forecasts. One of the first papers of
the UBC group was published by
`Tangang et al. (1997) <https://link.springer.com/article/10.1007/s003820050156>`_.

More recently, researchers at the Utrecht University (UU) in the Netherlands used
ANNs to forecast the ENSO. An early attempt was made by
`Feng et al. (2016) <https://www.geosci-model-dev-discuss.net/gmd-2015-273/>`_
to use machine learning (ML) methods for the ENSO forecasts. However, this
research did not pass the peer-review. Feng et al. (2016) aimed to build a python
package, called `ClimateLearn <https://github.com/Ambrosys/climatelearn>`_. However,
it was not clear which aim ClimateLearn pursued and how it would contribute to the
research on ENSO prediction.

Based on the initial attempts in Feng et al. (2016),
`Nooteboom et al. (2018) <https://www.earth-syst-dynam.net/9/969/2018/>`_
developed a hybrid model which is a combination out of an Autoregressive
integrated moving average (ARIMA) and an ANN model.

***********************
The aim of this package
***********************

The **issue of already existing statisitcal models** is that it can be
difficult and time consuming to build up on them because:

· the code difficult to access

· the code is not easily transferable to other research (particular coding
style, different programming language, etc.)

· the research uses differing conventions, i.e. defintion for lead time

· accessing the used data sources and postprocessing the data is time
consuming

The research framework Ninolearn aims to tackle these issues. The framework is
initiated to **facilitate** collaboration, **speed up** the start up of research
and make realized research **more transparent, comparable and reproducable**.

************************
How does NinoLearn work?
************************

NinoLearn aims to automatize and accurately separate the steps that are involved
within the development process for a statisical model for statisitcal ENSO forecasts:

1. Download data
2. Read data
3. Clean data, harmonize data from different sources
4. Postprocess data
5. Build a statistical model
6. Training the model (following the best practice of a 3-split of the data set into a train, validation and test data set)
7. Evaluate the model (using standardized tests)

Point 6 and 7 are not included in a standardized way in the current version of
NinoLearn.

Download
========

At the start of the development of a statistical model,
one needs to download data from potentially multiple sources. Most often it can
be time consuming to find the correct source and write own downloading routines.

The module :mod:`ninolearn.download` provides routines that make the download
process for various data sources (e.g. NCEP reanalysis, ORAS4 data set, Warm
Water volume index, Oceanic Niño Index, etc.)
possible within a few lines.

In example for the download of the sea surface temparture (SST)
data from the ERSSTv5 data set as well as the ONI:

.. code-block:: python
    :linenos:

    from ninolearn.download import download, sources
    download(sources.SST_ERSSTv5)
    download(sources.ONI)

The downloaded data is directly saved into the raw data (*rawdir*) direcotory
that is specified in :mod:`ninolearn.pathes`.

Data preparation
================

Furthermore, the module :mod:`ninolearn.postprocess.prepare` provides the user
with methods to prepare the data such that all postprocessed data sets follow
the same conventions regarding i.e. the time axis format.

By simply executing

.. code-block:: python
    :linenos:

    from ninolearn.postprocess.prepare import prep_oni
    prep_oni()

The downloaded raw data file for the ONI is assigned with a practicable time axis
which is used for all postprocessed data. Moreover, the prepared data is directly
saved into the postprocessed data direcotory (*postdir*) that is specified in
:mod:`ninolearn.pathes`.

Postprocessing
==============

Some more postprocessing methods and classes are available in modules in the
sub-package :py:mod:`ninolearn.postprocess`.


Computing anomalies (:mod:`ninolearn.postprocess.anomaly`) and regriding data to
a common grid (currently a 2.5°x2.5° grid, :mod:`ninolearn.postprocess.regrid`)
is as easy as in the following code snippet for the SST data set from the
ERSSTv5:

.. code-block:: python
    :linenos:

    from ninolearn.IO import read_raw
    from ninolearn.postprocess.anomaly import postprocess
    from ninolearn.postprocess.regrid import to2_5x2_5

    sst_ERSSTv5 = read_raw.sst_ERSSTv5()
    sst_ERSSTv5_regrided = to2_5x2_5(sst_ERSSTv5)
    postprocess(sst_ERSSTv5_regrided)

The method :func:`ninolearn.postprocess.anomaly.postprocess` saves the initial
raw data file to the postprocessed data directory and renames it following a
naming convention that makes it easy to access later. Furthermore, seasonal
anomalies based on the reference period 1981-2010 are computed and the anomlies
are as well saved to the postprocessed data directory following the naming
convention.

In addition, more advanced postprocessing methids such as principal
component analysis (:mod:`ninolearn.postprocess.pca`) and (evolving) complex networks
(:mod:`ninolearn.postprocess.network`) are provided to facilitate the use of
these methods with the data set of interest.

Read data
=========

In addition, reading methods in :mod:`ninolearn.IO.read_raw` are available that
make it easy to read the raw data (not postprocessed) without the need to specify
all the details about the raw data set e.g. type of method to use to read the
raw data or number of header lines. For instance one can read the raw file
for the ONI as follows:

.. code-block:: python
    :linenos:

    from ninolearn.IO import read_raw
    oni_raw = read_raw.oni()

For the postprocessed data, a the :class:`ninolearn.IO.read_post.data_reader`
makes it easy to access the postprocessed data in a dynamic way, i.e. selecting
specific time windows and areas from the desired data set. For the ONI and the
SST anomaly (SSTA) form the ERSSTv5 this looks as follows:

.. code-block:: python
    :linenos:

    from ninolearn.IO.read_post import data_reader

    reader = data_reader(startdate='1980-01', enddate='2017-12',
                         lon_min=30, lon_max=280,
                         lat_min=-30, lat_max=30)

    oni = reader.read_csv('oni')
    sst = reader.read_netcdf('sst', dataset='ERSSTv5', processed='anom')

The ONI and the SSTA are now read for the same time period (January 1980 till
December 2017) and the SSTA for the specified regions (boundaries in degrees East).

Machine Learning
================

For the training and evaluation of machine learning models, NinoLearn aims to
standardize the corresponding procedures.

New models within NinoLearn need to be set up in a specific way, such that
they can be trained and evaluated following standardized procedures. At the moment,
this is still work in progress and just some parts of the code are brought into
a user friendly style.

The module :mod:`ninolearn.learn.models` already contains some models that were
developed during the Master Thesis of Petersik (2019)`. In particular
a Deep Ensemble (:mod:`ninolearn.learn.models.dem`) and an Encoder-Decoder
(:mod:`ninolearn.learn.models.encoderDecoder`) model is available.

The modlue :mod:`ninolearn.learn.fit` contains methods for a standardized
training of the model and the corresponding prediction. In the module
:mod:`ninolearn.learn.evaluation`, some methods are gathered to evaluate
models for on the entire time series as well as on different seasons and decades
using the RMSE and the Pearson correlation.

****************
Cited literature
****************
Tangang, F. T., Hsieh, W. W., & Tang, B. (1997). Forecasting the equatorial
Pacific sea surface temperatures by neural network models.
Climate Dynamics, 13(2), 135-147.

Feng, Q. Y., Vasile, R., Segond, M., Gozolchiani, A., Wang, Y., Abel, M., ... &
Dijkstra, H. A. (2016). ClimateLearn: A machine-learning approach for climate
prediction using network measures. Geoscientific Model Development.

Nooteboom, P. D., Feng, Q. Y., López, C., Hernández-García, E., & Dijkstra,
H. A. (2018). Using network theory and machine learning to predict El Niño,
Earth Syst. Dynam., 9, 969–983.