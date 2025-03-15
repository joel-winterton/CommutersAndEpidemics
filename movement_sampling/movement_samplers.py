import numpy as np


class Sampler:
    """
    Baseplate sampler, infection simulation should rely on methods from this class, and each type of sampler should implement these methods.
    """

    def __init__(self, od_matrix, population_sizes):
        """
        """
        # Fix diagonals to contain all population that are not commuting
        commuter_counts = od_matrix.sum(axis=1) - od_matrix.diagonal()
        non_commuters = population_sizes - commuter_counts
        np.fill_diagonal(od_matrix, non_commuters)
        self.od_matrix = od_matrix
        self.number_of_patches = population_sizes.size

    def sample(self, i):
        """
        Sample k (number of patches) one-hot vectors of $N_i$ individuals travelling from $i$ to $1<=j<=k$ from distribution pi_{ij}.
        :param i:
        :return:
        """
        pass


class RandomCommuterSampler(Sampler):
    """
    Random commuter sampler, individuals are chosen without replacement (kinda, they're shuffled since we want
    individuals to only commute to one place).
    """

    def __init__(self, od_matrix, population_sizes):
        super().__init__(od_matrix, population_sizes)

    def sample(self, i):
        """
        Return one-hot of just commuters going between $i$ and $j$.
        :param i:
        :return:
        """
        commuters = self.od_matrix[i, :]
        bins = np.arange(len(commuters))
        assignments = np.repeat(bins, commuters)
        np.random.shuffle(assignments)
        indicator_arrays = (np.eye(self.number_of_patches, dtype=int)[assignments]).T
        return indicator_arrays


class PerfectCommuterSampler(Sampler):
    """
    Perfect commuter sampler.
    """

    def __init__(self, od_matrix, population_sizes):
        super().__init__(od_matrix, population_sizes)
        self.od_matrix = od_matrix
        self.number_of_patches = od_matrix.shape[0]

        self.od_pointers = np.zeros(shape=(*od_matrix.shape, 2), dtype=int)
        self.one_hot_sizes = np.zeros(shape=self.number_of_patches, dtype=int)

        self.assign_commuters()

    def assign_commuters(self):
        """"
        Assign indexes for each commuter, which can then be used in one-hot encoding.
        """
        for i in range(self.number_of_patches):
            locations = self.od_matrix[i, :]
            size = locations.sum()
            cumulative = np.cumsum(locations)
            right_index = cumulative
            left_index = np.concatenate([[0], cumulative])[:-1]
            self.one_hot_sizes[i] = size
            self.od_pointers[i, :, 1] = right_index
            self.od_pointers[i, :, 0] = left_index

    def sample(self, i):
        """
        Return one-hot of just commuters going between $i$ and $j$.
        :param i:
        :param j:
        :return:
        """
        one_hot = np.zeros(shape=(self.number_of_patches, self.one_hot_sizes[i]), dtype=int)
        for i, (l, r) in enumerate(self.od_pointers[i, :, :]):
            one_hot[i, l:r] = 1
        return one_hot


class CommuterAndRandos(Sampler):
    pass
