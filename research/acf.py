# -*- coding: utf-8 -*-

from statsmodels.graphics.tsaplots import plot_acf

from ninolearn.IO.read_post import data_reader

reader = data_reader()

nino = reader.read_csv('nino34')

plot_acf(nino)