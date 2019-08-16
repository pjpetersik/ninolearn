from os.path import join
import pandas as pd
import xarray as xr
import numpy as np
from ninolearn.pathes import rawdir, postdir
from datetime import datetime

from ninolearn.postprocess.prepare import prep_other_forecasts

prep_other_forecasts()