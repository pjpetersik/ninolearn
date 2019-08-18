"""
This module contains a collection of pathes which are used within NinoLearn.

NOTE: Specifiy the datadir in a private module which you may not commit to
you public repository
"""

from os.path import join

try:
    from ninolearn.private import datadir
except ImportError:
    raise ImportError("Cannot import name 'datadir'. Specifiy the path to your data directory using the name 'datadir' in the  ninolearn.private module which you may not commit to you public repository")

rawdir = join(datadir, 'raw')
processeddir = join(datadir, 'processed')
modeldir = join(datadir, 'model')
ed_model_dir = join(datadir, 'ED_model')