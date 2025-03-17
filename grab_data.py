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


def get_matrix(dataset='CENSUS_LAD11'):
    """"
    Returns commuter matrix as numpy ndarray. Has 2011 LAD census and mock data.
    Read more about 2011 data: `2011_census_data/README.md`
    """
    res = mock_commuter_flow
    if dataset == 'CENSUS_LAD11':
        res = np.genfromtxt(os.path.join(BASE_DIR, 'datasets/2011_census/clean/od_matrix.csv'), delimiter=',')
    if dataset == 'CENSUS_GLOBAL':
        res = np.genfromtxt(os.path.join(BASE_DIR, 'datasets/2011_census/global_geography/od_matrix.csv'),
                            delimiter=',')
    if dataset == 'CENSUS_SUBSAMPLED':
        res = np.genfromtxt(os.path.join(BASE_DIR, 'datasets/subsamples_2011_census/global_geography/od_matrix.csv'),
                            delimiter=',')
    return res.astype(int)


def get_population_sizes(dataset='CENSUS_LAD11'):
    """"
    Returns population size numpy array. Has 2011 LAD census and mock data.
    Read more about 2011 data: `2011_census_data/README.md`
    """
    pop_sizes = mock_commuter_flow.sum(axis=1) + mock_non_commuter_counts
    if dataset == 'CENSUS_LAD11':
        pop_sizes = np.genfromtxt(os.path.join(BASE_DIR, 'datasets/2011_census/clean/population_counts.csv'),
                                  delimiter=',')
    if dataset == 'CENSUS_GLOBAL':
        pop_sizes = np.genfromtxt(os.path.join(BASE_DIR, 'datasets/2011_census/global_geography/population_counts.csv'),
                                  delimiter=',')
    if dataset == 'CENSUS_SUBSAMPLED':
        pop_sizes = np.genfromtxt(
            os.path.join(BASE_DIR, 'datasets/subsamples_2011_census/global_geography/population_sizes.csv'),
            delimiter=',')
    return pop_sizes.astype(int)


def get_population_ordering(dataset='CENSUS_LAD11'):
    """
    Returns an ordered list of patch labels, all data inside a dataset is sorted according to this list.
    :param dataset:
    :return:
    """
    if dataset == 'CENSUS_LAD11':
        return np.genfromtxt(os.path.join(BASE_DIR, 'datasets/2011_census/clean/lad_codes.csv'), delimiter=',',
                             dtype=str)
    if dataset == 'CENSUS_GLOBAL':
        return np.genfromtxt(os.path.join(BASE_DIR, 'datasets/2011_census/global_geography/lad_codes.csv'),
                             delimiter=',', dtype=str)
    if dataset == 'CENSUS_SUBSAMPLED':
        return np.genfromtxt(os.path.join(BASE_DIR, 'datasets/subsamples_2011_census/global_geography/lad_codes.csv'),
                             delimiter=',', dtype=str)
