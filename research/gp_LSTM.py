import numpy as np
from keras.layers import LSTM, GRU, SimpleRNN, Dense

from ninolearn.learn.evolution import SuperGenome, Population
from ninolearn.learn.rnn import Data, RNNmodel
from ninolearn.utils import print_header, small_print_header


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

# Define the SuperGenome
bp_dict = {'window_size': [1, 36],
           'n_neurons': [5, 50],
           'Dropout': [0.0, 0.5],
           'lr': [0.00001, 0.001],
           'batch_size': [1, 100],
           'es_epochs': [5, 50],
           'features': [['wwv',   'nino34', 'pca1', 'pca2', 'pca3',
                        'c2_air',  'c3_air', 'c5_air',
                        'S', 'H', 'tau', 'C', 'L'], 5]}

sg = SuperGenome(bp_dict)

# Set the evolution parameters
population_size = 30
survivors = 10
offsprings = 10
random = 10

generations = 10

# Initialize the population
p = Population(sg, size=population_size, n_survivors=survivors,
               n_offsprings=offsprings, n_random_new=random)

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


        data_obj.load_features(genome['features'])

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
    print()
    print(f"FITNESS: {fitness}")
    print()

    p.getFitness(np.array(fitness))
    p.survival_of_the_fittest()
    p.start_new_generation()

    print()
    print(f"SURVIVORS: ")
    print()
    for i in range(len(p.survivor_population)):
        print(p.survivor_population[i].genes)

    print()
    print(f"NEW POPULATION:")
    print()
    for i in range(len(p.new_population)):
        print(p.new_population[i].genes)