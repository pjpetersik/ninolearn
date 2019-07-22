<img src="https://github.com/pjpetersik/ninolearn/blob/master/logo/logo.png" width="250">

NinoLearn is a research framework for the application of machine learning (ML)
methods for the prediction of the El Nino-Southern Oscillation (ENSO).

It contains methods for downloading relevant data from their sources, reading
raw data, postprocessing it and then reading the postprocessed data.

Two ML-models are currently available for the ENSO forecasts:

· Deep Ensemble Model (DEM)
· Encoder-Decoder Ensemble Model

# Conventions
This repository follows the convention that any monthly data is assigned to the
first day of the corresponding month. Therefore, if one deals with seasonal data,
the data is assigned to the first day of the last month of the season, i.e.
data for DJF is assigned to the 1st of Feburary.

# Installation
Add the path off the ninolearn base directory to your PYTHONPATH.
