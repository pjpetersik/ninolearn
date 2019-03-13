from ninolearn.io import read_data


import numpy as np
import matplotlib.pyplot as plt

nino34 = read_data.nino34_anom().values
wwv = read_data.wwv_anom().values
sst = read_data.sst_ERSSTv5()