from ninolearn.IO.read_data import data_reader

import numpy as np
import matplotlib.pyplot as plt

reader = data_reader()
data = reader.nino34_anom()
data2 = reader.wwv_anom()
data3 = reader.sst_ERSSTv5()
data4 = reader.uwind()
data5 = reader.vwind()