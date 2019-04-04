import matplotlib.pyplot as plt
from ninolearn.IO.read_post import data_reader


reader = data_reader()

wwv = reader.read_csv('wwv')
nino = reader.read_csv("nino34")


plt.plot(wwv/max(wwv), label="wwv")
plt.plot(nino/max(nino), label="nino")
plt.legend()