"""
This runs an epidemic based on the movement sampling model.
Depends on movement sampler class.
"""
import numpy as np
from movement_samplers import Sampler


class Epidemic:
    def __init__(self, movement_sampler: type(Sampler), od_matrix, population_sizes,
                 seed_patch=0,
                 beta=1.8, gamma=0.3, psi=0.8, t_max=100):
        self.population_sizes = population_sizes
        self.movement_sampler = movement_sampler(od_matrix, population_sizes)
        self.od_matrix = od_matrix
        self.number_of_patches = int(population_sizes.shape[0])

        self.s = [np.full(shape=(t_max, self.population_sizes[i]), fill_value=1) for i in range(self.number_of_patches)]
        self.i = [np.zeros(shape=(t_max, self.population_sizes[i])) for i in range(self.number_of_patches)]
        self.r = [np.zeros(shape=(t_max, self.population_sizes[i])) for i in range(self.number_of_patches)]

        self.beta = beta
        self.psi = psi
        self.gamma = gamma
        self.t_max = t_max
        self.seed(seed_patch)

    def seed(self, patch_index):
        """
        Infect a single random individual in given patch at t=0.
        :param patch_index:
        :return:
        """
        individual = np.random.randint(0, self.population_sizes[patch_index])
        self.s[patch_index][0, individual] = 0
        self.i[patch_index][0, individual] = 1

    def exerted_foi_matrix(self, t):
        """
        Returns exerted FOI matrix.
        :param t:
        :return:
        """
        external_foi = np.zeros(shape=(self.number_of_patches, self.number_of_patches), dtype=float)
        # pre-generate sum of infection amounts
        for k in range(self.number_of_patches):
            infection_state = self.i[k][t, :]
            movement = self.movement_sampler.sample(k)
            external_foi[k, :] = np.dot(movement, infection_state)
        return external_foi

    def step(self, t):
        """
        Calculate s,i,r state at t+1 given state at t.
        :param t:
        :return:
        """
        if t >= self.t_max:
            return False

        # exerted_foi_matrix = self.exerted_foi_matrix(t)
        s_tot, i_tot, r_tot = (np.zeros(shape=self.number_of_patches, dtype=int),
                               np.zeros(shape=self.number_of_patches, dtype=int),
                               np.zeros(shape=self.number_of_patches, dtype=int))
        s_indices, i_indices = [], []
        forces = np.zeros(self.number_of_patches)
        for j in range(self.number_of_patches):
            s_tot[j] = self.s[j][t, :].sum()
            i_tot[j] = self.i[j][t, :].sum()
            r_tot[j] = self.r[j][t, :].sum()

        for j in range(self.number_of_patches):
            s_indices.append(np.argwhere(self.s[j][t, :] == 1))
            i_indices.append(np.argwhere(self.i[j][t, :] == 1))
            within = i_tot[j]
            between_hack = np.dot(self.od_matrix[j,:], i_tot/self.population_sizes) - self.od_matrix[j,j]*i_tot[j]/self.population_sizes[j]
            forces[j] = self.beta / self.population_sizes[j] * (self.psi * within + (1 - self.psi) * between_hack)


        probabilities = 1 - np.exp(-forces)
        rng = np.random.default_rng()

        infection_deltas = rng.binomial(s_tot, probabilities)
        recovery_deltas = rng.binomial(i_tot, 1 - np.exp(-self.gamma))
        # choose infection deltas from indices
        for j in range(self.number_of_patches):
            self.s[j][t + 1, :] = self.s[j][t, :]
            self.i[j][t + 1, :] = self.i[j][t, :]
            self.r[j][t + 1, :] = self.r[j][t, :]

            new_infections = rng.choice(s_indices[j], infection_deltas[j])
            new_recoveries = rng.choice(i_indices[j], recovery_deltas[j])
            # Update individual states
            self.s[j][t + 1, new_infections] = 0
            self.i[j][t + 1, new_infections] = 1
            self.i[j][t + 1, new_recoveries] = 0
            self.r[j][t + 1, new_recoveries] = 1
        return True

    def simulate(self):
        """
        Populate time states.
        :return:
        """
        for t in range(self.t_max - 1):
            print(f"Currently at step {t}", end='\r')
            self.step(t)
        print("Simulation finished, aggregating states.")
        return self.aggregate_states()

    def aggregate_states(self):
        """
        Aggregate individual states to give total SIR timeseries for each patch.
        :return:
        """
        s, i, r = np.zeros(shape=(self.number_of_patches, self.t_max), dtype=int), np.zeros(
            shape=(self.number_of_patches, self.t_max), dtype=int), np.zeros(shape=(self.number_of_patches, self.t_max),
                                                                             dtype=int)
        for t in range(self.t_max):
            for patch in range(self.number_of_patches):
                s[patch, t] = self.s[patch][t, :].sum()
                i[patch, t] = self.i[patch][t, :].sum()
                r[patch, t] = self.r[patch][t, :].sum()
        return s.T, i.T, r.T
