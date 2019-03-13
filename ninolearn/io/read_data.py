from os.path import join
import pandas as pd

from ninolearn.private import datadir

def nino34_anom():
    """
    get the Nino3.4 Index data
    """
    data = pd.read_csv(join(datadir,"nino34.txt"), delim_whitespace=True)
    return data.ANOM