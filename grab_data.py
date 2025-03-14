"""
This exposes a function that will give a numpy commuter matrix to avoid doing this many times in each analysis.
"""
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

mock_commuter_flow = np.array([
    [100, 10, 2],
    [15, 150, 10],
    [35, 5, 200]
], dtype=int)

mock_non_commuter_counts = np.array([25, 10, 40])


def get_matrix(dataset='LAD11'):
    """"
    Returns commuter matrix as numpy ndarray. Has 2011 LAD census and mock data.
    Read more about 2011 data: `2011_census_data/README.md`
    """
    if dataset == 'LAD11':
        return np.genfromtxt(os.path.join(BASE_DIR, 'datasets/2011_census/clean/od_matrix.csv'), delimiter=',')
    if dataset == 'BBC_LAD':
        return np.genfromtxt(os.path.join(BASE_DIR, 'datasets/bbc_compatible/clean/od_matrix.csv'), delimiter=',')

    if dataset == 'MOCK':
        return mock_commuter_flow


def get_population_sizes(dataset='LAD11'):
    """"
    Returns population size numpy array. Has 2011 LAD census and mock data.
    Read more about 2011 data: `2011_census_data/README.md`
    """
    if dataset == 'LAD11':
        return np.genfromtxt(os.path.join(BASE_DIR, 'datasets/2011_census/clean/population_counts.csv'), delimiter=',')
    if dataset == 'BBC_LAD':
        return np.genfromtxt(os.path.join(BASE_DIR, 'datasets/2011_census/bbc_compatible/population_counts.csv'), delimiter=',')

    if dataset == 'MOCK':
        return mock_commuter_flow.sum(axis=1) + mock_non_commuter_counts


def get_population_ordering(dataset='LAD11'):
    """
    Returns an ordered list of patch labels, all data inside a dataset is sorted according to this list.
    :param dataset:
    :return:
    """
    if dataset == 'LAD11':
        return np.genfromtxt(os.path.join(BASE_DIR, 'datasets/2011_census/clean/lad_codes.csv'), delimiter=',', dtype=str)
    if dataset == 'BBC_LAD':
        return np.genfromtxt(os.path.join(BASE_DIR, 'datasets/2011_census/bbc_compatible/lad_codes.csv'), delimiter=',', dtype=str)

