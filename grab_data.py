"""
This exposes a function that will give a numpy commuter matrix to avoid doing this many times in each analysis.
"""
import csv

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
    sources = {'CENSUS_LAD11': 'datasets/2011_census/clean/od_matrix.csv',
               'CENSUS_GLOBAL': 'datasets/2011_census/global_geography/od_matrix.csv',
               'CENSUS_SUBSAMPLED': 'datasets/subsamples_2011_census/global_geography/od_matrix.csv',
               'BBC_FURTHEST_GLOBAL': 'datasets/bbc_pandemic/global_geography/furthest/od_matrix.csv',
               'BBC_NEXT_GLOBAL': 'datasets/bbc_pandemic/global_geography/next/od_matrix.csv'}
    if dataset in sources:
        return np.genfromtxt(os.path.join(BASE_DIR, str(sources[dataset])), delimiter=',').astype(int)
    print('Dataset not found, using mock data.')
    return mock_commuter_flow


def get_population_sizes(dataset='CENSUS_LAD11'):
    """"
    Returns population size numpy array. Has 2011 LAD census and mock data.
    Read more about 2011 data: `2011_census_data/README.md`
    """
    sources = {'CENSUS_LAD11': 'datasets/2011_census/clean/population_counts.csv',
               'CENSUS_GLOBAL': 'datasets/2011_census/global_geography/population_counts.csv',
               'CENSUS_SUBSAMPLED': 'datasets/subsamples_2011_census/global_geography/population_sizes.csv',
               'BBC_NEXT_GLOBAL': 'datasets/2011_census/global_geography/population_counts.csv',
               'BBC_FURTHEST_GLOBAL': 'datasets/2011_census/global_geography/population_counts.csv',
               }
    if dataset in sources:
        return np.genfromtxt(os.path.join(BASE_DIR, str(sources[dataset])), delimiter=',').astype(int)
    print('Dataset not found, using mock data.')
    return mock_commuter_flow.sum(axis=1) + mock_non_commuter_counts


def get_population_ordering(dataset='CENSUS_LAD11'):
    """
    Returns an ordered list of patch labels, all data inside a dataset is sorted according to this list.
    :param dataset:
    :return:
    """
    sources = {'CENSUS_LAD11': 'datasets/2011_census/clean/lad_codes.csv',
               'CENSUS_GLOBAL': 'datasets/2011_census/global_geography/lad_codes.csv',
               'CENSUS_SUBSAMPLED': 'datasets/subsamples_2011_census/global_geography/lad_codes.csv',
               'BBC_NEXT_GLOBAL': 'datasets/bbc_pandemic/global_geography/ordering.csv',
               'BBC_FURTHEST_GLOBAL': 'datasets/bbc_pandemic/global_geography/ordering.csv'}

    if dataset in sources:
        with open(os.path.join(BASE_DIR, str(sources[dataset])), 'r') as f:
            reader = csv.reader(f)
            return np.array(list(reader)).flatten()
    print('Dataset not found, mock data does not need ordering.')
