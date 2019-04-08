import matplotlib.pyplot as plt
from ninolearn.IO.read_post import data_reader


reader = data_reader()

wwv = reader.read_csv('wwv')
nino34M = reader.read_csv("nino3.4M")
nino34S = reader.read_csv("nino3.4S")
nino3M = reader.read_csv("nino3M")
nino12M = reader.read_csv("nino1+2M")

nino34M.plot()
nino34S.plot()
nino3M.plot()
nino12M.plot()