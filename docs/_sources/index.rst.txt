.. ninolearn documentation master file, created by
   sphinx-quickstart on Thu Jul 18 11:31:57 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NinoLearn - A research framemork for statistical ENSO prediction
================================================================
NinoLearn is a open source research framework  for statistical ENSO prediction
that is initiated to **facilitate** collaboration, **speed up** the start up of
research and make realized research **more transparent, comparable and reproducable**.

The current framework was developed in 2019 during the Master Thesis of
Paul Petersik. It lays the groundwork for a further development and the integration
of other postprocessing routines and statistical models.


The El Niño Southern Oscillation
################################

The El Niño Southern Oscillation (ENSO) is a coupelled ocean atmosphere phenomenon
which is present in the equatorial Pacific and affects the weather around the
world. In its positive phase, temperatures throught the equatorial Pacific are
relatively warm (El Niño phase). The other way around, temperatures are realtively
low in the negative phase (La Niña).

ENSO forecasts
##############

The predictive horizon for ENSO forecasts is by far longer than
for weather forecasts, because of the strong autocorrelation of the ENSO for
time periods up to about 6-9 month. Hence, there is a great interest in
researchto make skillful seasonal forecasts for the ENSO. Whereas, dynamical
models integratephysical equations that determine the evolution of the system
in time, statistical models *learn* from past observations how the future state
of the system will likely evolve. You can find current forecasts from dynamical
and statisticalmodels on the website of the Internation Research Insitute for
Climate and Society (see `here <https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/>`_).

Existing statistical models
###########################

Multiple research statistical models for the ENSO predictions have been
developed in the past. For instance a working group around
`William Hsieh <https://www.eoas.ubc.ca/people/williamhsieh>`_ at the
University of British Columbia (UBC) in Canada investigated the application of artifical
neural networks (ANN) for the ENSO forecasts. One of the first papers of
the UBC group was published by
`Tangang et al. (1997) <https://link.springer.com/article/10.1007/s003820050156>`_.
More recently, researchers the Utrecht University (UU) in the Netherlands used
ANNs to forecast the ENSO. An early attempt was made by
`Feng et al. (2016) <https://www.geosci-model-dev-discuss.net/gmd-2015-273/>`_
to use machine learning (ML) methods for the ENSO forecasts. In this unpublished
research they aimed to build a python package for ENSO forecasts that is available
on Github (see `here <https://github.com/Ambrosys/climatelearn>`_). Based on
this initial attempt, `Nooteboom et al. (2018) <https://www.earth-syst-dynam.net/9/969/2018/>`_
developed a hybrid model which is a combination out of an Autoregressive
integrated moving average (ARIMA) and an ANN model.

The aim of this package
#######################

The **issue of already existing statisitcal models** is that it can be
difficult and time consuming to build up on them because:

· the code is  not easily accessible

· the code is not easily transferable to other research (particular coding
style, different programming language, etc.)

· the research uses differing conventions, i.e. defintion for lead time

· accessing the used data sources and postprocessing the data is time
consuming

The research framework Ninolearn aims to tackle these shortcomings. The framework
initiated to **facilitate** collaboration, **speed up** the start up of research
and make realized research **more transparent, comparable and reproducable**.

How does NinoLearn work?
#########################

NinoLearn aims to automatize various steps within the development process for
a statisical model.

At the start of the development of a statistical model,
one needs to download data from potentially multiple sources.
The module :mod:`ninolearn.download` provides routines that make the download
process for various data source (NCEP reanalysis, ORAS4 dataset, WWV index, ONI index, etc.)
a *one-liner*.

Furthermore, the module :mod:`ninolearn.postprocess.prepare` provides the user
with methods to prepare the data such that all postprocessed data sets follow
the same conventions regarding i.e. the time axis format.

In addition, reading methods in :mod:`ninolearn.IO.read_raw` are available that
make it easy to read the raw data (not postprocessed) without the need to specify
all the details about the raw data set e.g. type of method to use to read the
raw data or number of header lines.
For the postprocessed data, a the :class:`ninolearn.IO.read_post.data_reader`
makes it easy to access the postprocessed data in a dynamic way, i.e. selecting
specific time windows and areas from the desired data set.

Some more postprocessing methods and classes such as computing anomalies
(:mod:`ninolearn.postprocess.anomaly`), regriding data to a common grid
(currently a 2.5°x2.5° grid, :mod:`ninolearn.postprocess.regrid`), principal
component analysis (:mod:`ninolearn.postprocess.pca`) and (evolving) complex networks
(:mod:`ninolearn.postprocess.network`) are provided to facilitate the use of
these methods with the data set of interest.

Finally, some models that were developed during the Master Thesis of
Petersik (2019) are available in :mod:`ninolearn.learn.models`. In particular
a Deep Ensemble (:mod:`ninolearn.learn.models.dem`) and an Encoder-Decoder
(:mod:`ninolearn.learn.models.encoderDecoder`) model is available.



Contents
########
.. toctree::
   :maxdepth: 2

   install
   package
   tutorials
   forecasts

Literature
##########
Tangang, F. T., Hsieh, W. W., & Tang, B. (1997). Forecasting the equatorial
Pacific sea surface temperatures by neural network models.
Climate Dynamics, 13(2), 135-147.

Feng, Q. Y., Vasile, R., Segond, M., Gozolchiani, A., Wang, Y., Abel, M., ... &
Dijkstra, H. A. (2016). ClimateLearn: A machine-learning approach for climate
prediction using network measures. Geoscientific Model Development.

Nooteboom, P. D., Feng, Q. Y., López, C., Hernández-García, E., & Dijkstra,
H. A. (2018). Using network theory and machine learning to predict El Niño,
Earth Syst. Dynam., 9, 969–983.
