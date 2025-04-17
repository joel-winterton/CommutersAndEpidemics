"""
Modified EPR.
This is an EPR model but with timing of next jump decided by where we go (so we can control how long we stay at home and work).
"""
from typing import cast
import numpy as np

from lattice_model import Lattice, Coordinate, Cube
from trajectories.lattice_model import Offset


class Individual:
    lattice: Lattice

    def __init__(self, lattice: Lattice, seed: Coordinate, rho=0.3, gamma=0.21, exp_time=1 / 24, tau_h=9 / 24,
                 tau_w=8 / 24, steps=100):
        """
        lattice: Lattice to work on.
        rho, gamma: Parameters controlling probability of exploring.
        time_param: Parameter controlling exponential jump time distribution.
        tau_w, tau_h: Time to spend at work/home respectively.
        """
        self.lattice = lattice
        self.seed = seed
        self.steps = steps
        self.rho = rho
        self.gamma = gamma
        self.exp_time = exp_time
        self.tau_w = tau_w
        self.tau_h = tau_h
        self.visited_freq = dict()
        self.visited_freq[seed] = 1
        self.visiting_list = [(seed, 1)]
        self.space_trajectory = [seed]
        self.time_trajectory = [self.tau_h]

    def rank_trajectory(self):
        """
        Sort trajectory into list sorted in descending order of visitation frequency.
        Returned list is of format (Coordinate, freq(Coordinate))[]
        """
        return sorted(list(self.visited_freq.items()), key=lambda x: x[1], reverse=True)

    def visit(self, coord: Coordinate):
        """
        Should update trajectory and visitation frequencies,
        but before that it should decide how long individual is staying at this location.
        """
        # get 1st and 2nd most visited coords
        t_jump = 0
        if coord == self.visiting_list[0][0]:
            t_jump = self.tau_h
        elif len(self.visiting_list) > 1 and coord == self.visiting_list[1][0]:
            t_jump = self.tau_w
        # TODO add probability of drawing from exp in home/work case as well?
        else:
            t_jump = np.random.default_rng().exponential(scale=self.exp_time, size=1)[0]

        if coord in self.visited_freq:
            self.visited_freq[coord] += 1
        else:
            self.visited_freq[coord] = 1
        self.space_trajectory.append(coord)
        self.time_trajectory.append(t_jump)
        self.visiting_list = self.rank_trajectory()

    def simulate(self):
        for _ in range(self.steps):
            # This autonomously updates state (hence no t index), but is still dependent on prev state
            s = len(self.visited_freq)
            p_explore = self.rho * s ** (-self.gamma)
            uniform = np.random.default_rng().uniform()
            current_point = self.space_trajectory[-1]
            r, c = current_point.offset.array
            if uniform < p_explore:
                # explore
                prob = self.lattice.exploration_probabilities[r, c, ...]
                flat_index = np.random.choice(prob.size, p=prob.ravel())
                destination_r, destination_c = np.unravel_index(flat_index, prob.shape)
                destination = Coordinate(offset=Offset(destination_r, destination_c))
                self.visit(destination)
            else:
                # return
                coords = [item[0] for item in self.visiting_list]
                probs_prop = 1 / np.arange(1, len(self.visiting_list) + 1)
                # summing over this, really? Yes. I am tired.
                probabilities = probs_prop / probs_prop.sum()
                next_coord = np.random.choice(coords, size=1, p=probabilities).flatten()[0]
                self.visit(next_coord)
        return self.space_trajectory, self.time_trajectory


class Movement:
    """
    Wraps individual movement class, and does it on bulk.
    """

    def __init__(self, lattice_width=25, population_avg=100):
        self.lattice = Lattice(width=lattice_width, population_avg=population_avg)

    def simulate(self, steps=50):
        """
        Simulate trajectories, starting trajectories on generated lattice census.
        """
        trajectories = []
        for r in range(self.lattice.r):
            for c in range(self.lattice.c):
                pop = self.lattice.population[r, c]
                print(f"Simulating {pop} trajectories from {r},{c}")
                seed = Coordinate(offset=Offset(r, c))
                for i in range(pop):
                    person = Individual(lattice=self.lattice, seed=seed, steps=steps)
                    trajectories.append(person.simulate())
        return np.array(trajectories)

    def calculate_indicators(self, trajectories, delta, t_max):
        """
        Create array for each individual per time-step that gives their location.
        """
        steps = np.ceil(t_max / delta).astype(int)
        for i in trajectories:
            pass

