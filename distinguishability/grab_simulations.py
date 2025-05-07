"""
Grabs simulations based off required passed parameters. Saves having to load a lot of files each time.
"""
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

available_r0s = [1.3, 1.8, 2.0, 2.5, 3.0, 4.0]


def grab_simulations(r0, model_code):
    string = f'_model={model_code}, r0={r0}, incubation=3, max_time=200, time_period = 12.0, samples=50, shape=(50, 2400, 346).npy'
    directory = 'simulation_data/'
    if r0 not in available_r0s:
        raise ValueError('No simulation with that R0 available.')

    s_path, i_path = os.path.join(BASE_DIR, directory + 'S' + string), os.path.join(BASE_DIR, directory + 'I' + string)

    return np.load(s_path), np.load(i_path)


class AvailableSims:
    def __init__(self):
        self.current_index = -1
        self.r0s = available_r0s

    def __iter__(self):
        return self

    def __next__(self):
        self.current_index += 1
        if self.current_index < len(self.r0s) :
            one_way = grab_simulations(self.r0s[self.current_index], model_code='perfect_oneway')
            two_way = grab_simulations(self.r0s[self.current_index], model_code='perfect_twoway')
            return one_way, two_way, self.r0s[self.current_index]
        raise StopIteration
