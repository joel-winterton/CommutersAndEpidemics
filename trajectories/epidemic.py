"""
Class to run epidemic on trajectories.
Main idea: Compute a list of pointers to individual objects for each time frame.
"""
import numpy as np


class Epidemic:
    def __init__(self, lattice, trajectories, delta=1 / 24, t_max=80):
        self.lattice = lattice
        lattice_shape = self.lattice.grid.shape
        self.trajectories = trajectories
        # individual based indicators
        self.s, self.i, self.r = (np.full(len(trajectories), 1, dtype=int),
                                  np.zeros(len(trajectories), dtype=int),
                                  np.zeros(len(trajectories), dtype=int))
        steps = np.ceil(t_max / delta).astype(int)
        # site based indicators, structure: site_index: [individual pointer]
        self.site_s, self.site_i, self.site_r = [], [], []
        t = 0
        while t < t_max:
            # assume at previous time-step, we had all positions correct, so all we have to do is update trajectories
            # that have occurred in last delta

            t += delta
