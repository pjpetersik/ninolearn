import numpy as np

from ninolearn.utils import lowest_indices

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
            if SGinstance.bp_gene_type[name] != list:
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

            elif SGinstance.bp_gene_type[name] == list:
                # replace one entry by another from the blue print list that
                # is not already in the mutant genome
                mutant_dict[name] = self.genes[name].copy()

                i = np.random.randint(len(mutant_dict[name]))
                j =  np.random.randint(len(SGinstance.bp_genome[name][0]))

                for _ in range(len(SGinstance.bp_genome[name][0])):
                    if not SGinstance.bp_genome[name][0][j] in mutant_dict[name]:
                        j =  np.random.randint(len(SGinstance.bp_genome[name][0]))

                mutant_dict[name][i] = SGinstance.bp_genome[name][0][j]

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

        self.bp_gene_type = {}
        for name in self.gene_names:
            self.bp_gene_type[name] = type(self.bp_genome[name][0])

    def randomGenome(self):
        """
        Returns a random Gen instance.
        """
        genes = {}

        for name in self.gene_names:
            if self.bp_gene_type[name] == list:
                genes[name] = self.randomSelection(self.bp_genome[name])
            else:
                genes[name] = self.randomGeneValue(self.bp_genome[name])
        return Genome(genes)

    def randomSelection(self, blue_print_gene_list):
        """
        Selection a certain number of genes randomly from a list
        """
        assert type(blue_print_gene_list) == list
        assert type(blue_print_gene_list[0]) == list
        assert type(blue_print_gene_list[1]) == int

        selection = np.random.choice(blue_print_gene_list[0],
                                     size=blue_print_gene_list[1],
                                     replace=False).tolist()
        return selection

    def randomGeneValue(self, blue_print_gene_value):
        """
        Returns a random value from the blue print segmant value using
        a uniform distribution.
        """
        assert type(blue_print_gene_value) is list

        # for float number use continuous  uniform distribution
        if type(blue_print_gene_value[0]) is float:
            value = np.random.uniform(low=blue_print_gene_value[0],
                                      high=blue_print_gene_value[1])

        # for discrete numbers use discrete unifrom distribution
        elif type(blue_print_gene_value[0]) is int:
            value = np.random.randint(low=blue_print_gene_value[0],
                                      high=blue_print_gene_value[1] + 1)
        return value


class Population(object):
    """
    A population is a collection of collection of multiple genomes.
    """
    def __init__(self, superGenomeInstanace, size = 15, n_survivors = 5,
                 n_offsprings=5, n_random_new = 5):
        """

        :type size: int
        :param size: the number of Genomes in a population

        :param superGenomeInstanace: An instance of the SuperGenome class.
        """
        assert size == n_survivors + n_offsprings + n_random_new

        self.size = size
        self.n_survivors = n_survivors
        self.n_offsprings = n_offsprings
        self.n_random_new = n_random_new

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

    def survival_of_the_fittest(self):
        """
        Let evolution decide which model survives and can later generate clones
        and offsprings.

        :param size: The number of survivors based on the score. At the moment
        a low score means a high fitness.
        """
        self.survivor_population = []

        il = lowest_indices(self.fitness, self.n_survivors)[0]
        for i in il:
            self.survivor_population.append(self.population[i])

    def makeNewPopulation(self):
        """
        Generate a new population by cloning, pairing and mutation as well as
        random new genomes
        """
        self.new_population = []

        self.make_clones()
        assert id(self.new_population) != id(self.survivor_population)

        self.make_offsprings()
        self.make_new_Genome()

    def make_clones(self):
        """
        Clone and mutate the survivor population.
        """
        assert self.n_survivors == len(self.survivor_population)

        for i in range(self.n_survivors):
            clone_genome = self.survivor_population[i].mutant(self.superGenome)
            self.new_population.append(clone_genome)

    def make_offsprings(self):
        """
        Make some offsprings from the surviror population and append them to
        the new population list.
        """
        for i in range(self.n_offsprings):
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

    def make_new_Genome(self):
        """
        Make some totally new Genomes.
        """
        for i in range(self.n_random_new):
            self.new_population.append(self.superGenome.randomGenome())

    def start_new_generation(self):
        """
        Overwrite the old population list with the new population.
        """
        self.makeNewPopulation()
        self.population = self.new_population.copy()