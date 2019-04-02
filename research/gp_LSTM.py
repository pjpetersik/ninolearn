import matplotlib.pyplot as plt
import numpy as np
from keras.layers import LSTM, GRU, SimpleRNN, Dense

from ninolearn.learn.rnn import Data, RNNmodel
from ninolearn.plot.evaluation import (plot_explained_variance,
                                       plot_correlations)
from ninolearn.utils import lowest_indices, print_header, small_print_header


class Genome(object):
    """
    A genome contains all information to generate one particular model out of
    it.
    """
    def __init__(self, genes_dict):
        """
        :type genes_dict: dict
        :param genes_dict: A dictionary with the name of a model parameter as
        key and the corresponding value as item.
        """
        assert type(genes_dict) is dict
        self.genes = genes_dict
        self.gene_names = genes_dict.keys()

    def __getitem__(self, key):
        return self.genes[key]

    def mutant(self, SGinstance, strength=0.02):
        """
        Returns a mutant genome of this genome.

        :param SGinstance: a SuperGenome instance to make sure that the mutant
        still correspons with the ranges given in the blue print genome

        :param strength: The strength of the mutation. Value must be greater 0
        and smaller equal 1. A strength of 1 means that the maximum possible
        mutation corresponds to the range between the low and the high value
        provided in the SuperGeneome instance
        """
        assert isinstance(SGinstance, SuperGenome)
        assert strength <= 1 and strength > 0
        mutant_dict = {}

        for name in self.gene_names:

            # calculate the strength of the mutation
            maxmutation = (SGinstance.bp_genome[name][1] -
                           SGinstance.bp_genome[name][0])

            mutation = maxmutation * np.random.uniform(low=-strength,
                                                       high=strength)

            # apply the mutation to the corresponding gene
            if type(self.genes[name]) == float:
                mutant_dict[name] = self.genes[name] + mutation

            elif type(self.genes[name]) == int:
                mutant_dict[name] = int(self.genes[name] + mutation)

            # make sure that the mutated gene stays in the correct range
            mutant_dict[name] = max(mutant_dict[name],
                                    SGinstance.bp_genome[name][0])
            mutant_dict[name] = min(mutant_dict[name],
                                    SGinstance.bp_genome[name][1])

        return Genome(mutant_dict)


class SuperGenome(object):
    """
    This is a sort of blue print gene. That can generate proper gene.
    """
    def __init__(self, blue_print_genome):
        """
        :param blue_print_segment_dictionary: A dictionary with keys
        corresponding to the name of a segment and the range of possible values
        as item.
        """
        assert type(blue_print_genome) is dict

        self.n_genes = len(blue_print_genome)
        self.gene_names = blue_print_genome.keys()
        self.bp_genome = blue_print_genome

    def randomGenome(self):
        """
        Returns a random Gen instance.
        """
        genes = {}

        for name in self.gene_names:
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


class Population(object):
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

        self._makeInitialPopulation()

    def _makeInitialPopulation(self):
        """
        Generate an initial population that is a list of Genome objects.
        """
        self.population = []

        for _ in range(self.size):
            self.population.append(self.superGenome.randomGenome())

    def getFitness(self, fitnessScore):
        """
        Get the fitness scores, that need to be calculated outside of the the
        class.

        :type fitnessScore: np.ndarray
        :param fitnessScore: A np.ndarray with the fitness score of each
        Genome.
        """

        assert type(fitnessScore) == np.ndarray
        self.fitness = fitnessScore

    def survival_of_the_fittest(self, size=2):
        """
        Let evolution decide which model survives and can later generate clones
        and offsprings.

        :param size: The number of survivors based on the score. At the moment
        a low score means a high fitness.
        """
        self.survivor_population = []

        il = lowest_indices(self.fitness, size)[0]
        for i in il:
            self.survivor_population.append(self.population[i])

    def makeNewPopulation(self, offsprings=2, random=2):
        """
        Generate a new population by cloning, pairing and mutation as well as
        random new genomes
        """
        self.new_population = []

        self.make_clones()
        assert id(self.new_population) != id(self.survivor_population)

        self.make_offsprings(offsprings)
        self.make_new_Genome(random)

    def make_clones(self):
        """
        Clone and mutate the survivor population.
        """
        for i in range(len(self.survivor_population)):
            clone_genome = self.survivor_population[i].mutant(self.superGenome)
            self.new_population.append(clone_genome)

    def make_offsprings(self, number):
        """
        Make some offsprings from the surviror population and append them to
        the new population list.
        """
        for i in range(number):
            off_spring_genomes = self._copulation(self.survivor_population)
            self.new_population.append(off_spring_genomes)

    def _copulation(self, population):
        """
        Let the entire population have some GREAT fun. The pure offsprings that
        are generated just by a random selection of genes from the population
        are mutated afterwards.
        """
        offspring_genome = {}

        for name in self.superGenome.gene_names:
            parent = np.random.randint(len(population))
            offspring_genome[name] = population[parent].genes[name]

        return Genome(offspring_genome).mutant(self.superGenome)

    def make_new_Genome(self, number):
        """
        Make some totally new Genomes.
        """
        for i in range(number):
            self.new_population.append(self.superGenome.randomGenome())

    def start_new_generation(self):
        """
        Overwrite the old population list with the new population.
        """
        self.population = self.new_population.copy()



if __name__ == "__main__":
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

    lead_time = 6

    generations = 10

    bp_dict = {'window_size': [1, 36],
               'n_neurons': [5, 50],
               'Dropout': [0.0, 0.5],
               'lr': [0.00001, 0.001],
               'batch_size': [1, 100],
               'es_epochs': [5, 50]
               }
    sg = SuperGenome(bp_dict)

    population_size = 8
    p = Population(population_size, sg)

    for n in range(generations):
        print_header(f" Generation Nr. {n} ")

        i = 0
        fitness = []
        for genome in p.population:
            small_print_header(f"Genome Nr. {i}")
            print(genome.genes)
            window_size = genome['window_size']
            n_neurons = genome['n_neurons']
            Dropout = genome['Dropout']
            lr = genome['lr']
            batch_size = genome['batch_size']
            es_epochs = genome['es_epochs']

            data_obj = Data(label_name="nino34", data_pool_dict=pool,
                            window_size=window_size, lead_time=lead_time,
                            startdate='1980-01', train_frac=0.6)


            data_obj.load_features(['wwv',  # 'nino34',
                                    #'pca1', 'pca2', 'pca3',
                                    'c2_air',  'c3_air', 'c5_air',
                                    'S', 'H', 'tau', 'C', 'L'
                                    ])

            model = RNNmodel(data_obj, Layers=[LSTM], n_neurons=[n_neurons],
                             Dropout=Dropout, lr=lr, epochs=500,
                             batch_size=batch_size, es_epochs=es_epochs,
                             verbose=0)

            model.fit()

            print(f"Stopped after {model.history.epoch[-1]} epochs")

            model.predict()

            trainRMSE, trainNRMSE = model.get_scores('train')
            testRMSE, testNRMSE = model.get_scores('test')
            shiftRMSE, shiftNRMSE = model.get_scores('shift')

            print('Train Score: %.2f MSE, %.2f NMSE' % (trainRMSE**2, trainNRMSE))
            print('Test Score: %.2f MSE, %.2f NMSE' % (testRMSE**2, testNRMSE))
            print('Shift Score: %.2f MSE, %.2f NMSE' % (shiftRMSE**2, shiftNRMSE))

            fitness.append(testNRMSE)
            i += 1

            #del model
            #del data_obj

        print("############################")

        print(f"FITNESS :{fitness}")
        print()

        p.getFitness(np.array(fitness))

        p.survival_of_the_fittest(size=2)
        p.makeNewPopulation(offsprings=4, random=2)
        p.start_new_generation()

        for i in range(len(p.survivor_population)):
            print(p.survivor_population[i].genes)

        print()

        for i in range(len(p.new_population)):
            print(p.new_population[i].genes)