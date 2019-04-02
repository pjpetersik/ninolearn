import matplotlib.pyplot as plt
import numpy as np
from keras.layers import LSTM, GRU, SimpleRNN, Dense

from ninolearn.learn.rnn import Data, RNNmodel
from ninolearn.plot.evaluation import (plot_explained_variance,
                                       plot_correlations)
from ninolearn.utils import lowest_indices

pool = {'c2_air': ['network_metrics', 'fraction_clusters_size_2', 'air_daily',
                   'anom', 'NCEP'],
        'c3_air': ['network_metrics', 'fraction_clusters_size_3', 'air_daily',
                   'anom', 'NCEP'],
        'c5_air': ['network_metrics', 'fraction_clusters_size_5', 'air_daily',
                   'anom', 'NCEP'],
        'tau': ['network_metrics', 'global_transitivity', 'air_daily', 'anom',
                'NCEP'],
        'C': ['network_metrics', 'avelocal_transmissivity', 'air_daily',
              'anom', 'NCEP'],
        'S': ['network_metrics', 'fraction_giant_component', 'air_daily',
              'anom', 'NCEP'],
        'L': ['network_metrics', 'average_path_length', 'air_daily', 'anom',
              'NCEP'],
        'H': ['network_metrics', 'hamming_distance', 'air_daily', 'anom',
              'NCEP'],
        'Hstar': ['network_metrics', 'corrected_hamming_distance', 'air_daily',
                  'anom',
                  'NCEP'],
        'nino34': [None, None, 'nino34', 'anom', None],
        'wwv': [None, None, 'wwv', 'anom', None],
        'pca1': ['pca', 'pca1', 'air', 'anom', 'NCEP'],
        'pca2': ['pca', 'pca2', 'vwnd', 'anom', 'NCEP'],
        'pca3': ['pca', 'pca2', 'uwnd', 'anom', 'NCEP'],
        }

# before evolution
lead_time = 3


class Genome(object):
    def __init__(self, genes_dict):
        self.genes = genes_dict

    def __getitem__(self, key):
        return self.genes[key]


class SuperGenome(object):
    """
    This is a sort of blue print gene. That can generate proper gene
    """
    def __init__(self, blue_print_genome):
        """
        :param blue_print_segment_dictionary: A dictionary with keys
        corresponding to the name of a segment and the range of possible values
        as item.
        """
        assert type(blue_print_genome) is dict

        self.n_genes = len(blue_print_genome)
        self.genes_names = blue_print_genome.keys()
        self.bp_genome = blue_print_genome

    def randomGenome(self):
        """
        Returns a random Gen instance.
        """
        genes = {}

        for name in self.genes_names:
            genes[name] = self.randomGeneValue(self.bp_genome[name])
        return Genome(genes)

    def randomGeneValue(self, blue_print_gene_value):
        """
        Returns a random value from the blue print segmant value using
        a uniform distribution.
        """
        assert type(blue_print_gene_value) is list

        if type(blue_print_gene_value[0]) is float:
            value = np.random.uniform(low=blue_print_gene_value[0],
                                      high=blue_print_gene_value[1])

        elif type(blue_print_gene_value[0]) is int:
            value = np.random.randint(low=blue_print_gene_value[0],
                                      high=blue_print_gene_value[1] + 1)
        return value


class poplulation(object):
    """
    A population is a collection of collection of multiple genomes.
    """
    def __init__(self, size, superGenomeInstanace):
        """

        :type size: int
        :param size: the number of Genomes in a population

        :param superGenomeInstanace: An instance of the SuperGenome class.
        """
        self.size = size

        assert isinstance(superGenomeInstanace, SuperGenome)
        self.superGenome = superGenomeInstanace

        self.makePopulation()

    def makePopulation(self):
        self.population = []

        for _ in range(self.size):
            self.population.append(self.superGenome.randomGenome())

    def getFitness(self, fitnessScore):
        """
        :param fitnessScore: A np.array with the fitness score of each Genome.
        """

        assert type(fitnessScore) == np.ndarray
        self.fitness = fitnessScore

    def survivors(self):
        """
        Genrate a new  population based on the fitness score
        """
        self.survivor_population = []

        il = lowest_indices(self.fitness, 2)[0]
        for i in il:
            self.survivor_population.append(self.population[i])



bp_dict = {'window_size': [6, 36],
           'n_neurons': [5, 50],
           'Dropout': [0.0, 0.5],
           'lr': [0.00001, 0.001],
           'batch_size': [1, 100],
           'es_epochs': [5, 100]
           }

sg = SuperGenome(bp_dict)
pop = poplulation(5, sg)
fitness = np.array([0.1,0.6,0.5,0.4,0.2])
pop.getFitness(fitness)
pop.survivors()


#%%
# the gene:
window_size = 6
n_neurons = 10
Dropout = 0.2
lr = 0.01
batch_size = 100
es_epochs = 20


LAYER = LSTM
data_obj = Data(label_name="nino34", data_pool_dict=pool,
                window_size=window_size, lead_time=lead_time,
                startdate='1980-01', train_frac=0.6)


data_obj.load_features(['wwv',  # 'nino34',
                        'pca1', 'pca2', 'pca3',
                        'c2_air',  'c3_air', 'c5_air',
                        'S', 'H', 'tau', 'C', 'L'
                        ])

model = RNNmodel(data_obj, Layers=[LSTM], n_neurons=[10], Dropout=0.0,
                 lr=0.0001, epochs=500, batch_size=100, es_epochs=20)

model.fit()
model.predict()

trainRMSE, trainNRMSE = model.get_scores('train')
testRMSE, testNRMSE = model.get_scores('test')
shiftRMSE, shiftNRMSE = model.get_scores('shift')

print('Train Score: %.2f MSE, %.2f NMSE' % (trainRMSE**2, trainNRMSE))
print('Test Score: %.2f MSE, %.2f NMSE' % (testRMSE**2, testNRMSE))
print('Shift Score: %.2f MSE, %.2f NMSE' % (shiftRMSE**2, shiftNRMSE))

# %%

plt.close("all")
model.plot_history()
model.plot_prediction()

plot_explained_variance(model.testY, model.testPredict[:, 0], model.testYtime)
plt.title(f"Lead time: {model.Data.lead_time} month")

plot_correlations(model.testY, model.testPredict[:, 0], model.testYtime)
# -*- coding: utf-8 -*-

