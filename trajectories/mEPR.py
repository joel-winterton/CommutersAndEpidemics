"""
Modified EPR.
This is an EPR model but with timing of next jump decided by where we go (so we can control how long we stay at home and work).
"""

import numpy as np

from lattice_model import Lattice, Coordinate
from truncated_levy_distribution import TruncatedLevy


class Individual:
    lattice: Lattice

    def __init__(self, lattice: Lattice, seed: Coordinate, rho=0.6, gamma=0.21):
        self.lattice = lattice
        self.seed = seed
        self.clock = 0
        self.rho = rho
        self.gamma = gamma
        self.length_distribution = TruncatedLevy(beta=0.9, k=25)
        self.time_distribution = TruncatedLevy(beta=0.5, k=17, x0=0)
        self.visited_freq = dict()
        self.visited_freq[seed] = 1
        self.space_trajectory = [seed]
        self.time_trajectory = [0]


class Person:
    lattice: Lattice

    def __init__(self, lattice: Lattice, seed: Coordinate, rho=0.6, gamma=0.21):
        self.lattice = lattice
        self.seed = seed
        self.clock = 0
        self.rho = rho
        self.gamma = gamma
        self.length_distribution = TruncatedLevy(beta=0.9, k=25)
        self.time_distribution = TruncatedLevy(beta=0.5, k=17, x0=0)
        self.visited_freq = dict()
        self.visited_freq[seed] = 1
        self.space_trajectory = [seed]
        self.time_trajectory = [0]

    def visit(self, origin, destination, time):
        """
        Visit a location and do the bookkeeping.
        :param origin:
        :param coords: Coordinates of the location to visit.
        :return:
        """
        if destination in self.visited_freq:
            self.visited_freq[destination] += 1
        else:
            self.visited_freq[destination] = 1
        self.space_trajectory.append(destination)

    def rank_trajectory(self):
        """
        Sort trajectory into list sorted in descending order of visitation frequency.
        """
        return sorted(list(self.visited_freq.items()), key=lambda x: x[1], reverse=True)

    def preferential_return(self, time):
        """
        Return to a known location according to Zipfs law, bookkeeping included.
        :return: Coordinate that has been returned to.
        """
        # Sort visits into descending order
        items = self.rank_trajectory()
        coords = [item[0] for item in items]

        probs_prop = 1 / np.arange(1, len(items) + 1)
        probabilities = probs_prop / probs_prop.sum()

        next_coord = np.random.choice(coords, size=1, p=probabilities).flatten()[0]
        self.visit(self.space_trajectory[-1], next_coord, time)
        return next_coord

    def explore(self, time):
        """
        Explore a new location.
        :return:
        """
        step_size = int(self.length_distribution.rvs(size=1))
        patches = self.lattice.ball(self.space_trajectory[-1], step_size)
        index = 0
        # This isn't great, since it means the jump is outside the geometry, try and avoid
        while len(patches) == 0:
            step_size = int(self.length_distribution.rvs(size=1))
            patches = self.lattice.ball(self.space_trajectory[-1], step_size)

        return np.random.choice(patches)

    def burn_in(self):
        """
        Burn in.
        """
        pass

    def run(self, time_steps=50):
        t = 0
        next_jump = 0
        for _ in range(time_steps):
            # loop through time more efficiently
            if t >= next_jump:
                uniform = np.random.default_rng().uniform()
                # calculate p_explore
                p_explore = self.rho * len(self.visited_freq.keys()) ** -self.gamma
                if uniform < p_explore:
                    next_coord = self.explore(next_jump)
                else:
                    next_coord = self.preferential_return(next_jump)
                # TODO: determine next time-step
            t += 1
