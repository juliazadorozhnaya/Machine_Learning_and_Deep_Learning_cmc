import numpy as np


def evaluate_measures(sample):
    sample = np.array(np.sort(sample))
    unique_elem, values = np.unique(sample, return_counts=True)
    new_pattern = values / np.sum(values)
    measures = {'gini': float(1 - np.sum(new_pattern ** 2)),
                'entropy': float((-1) * np.sum(new_pattern * np.log(new_pattern))),
                'error': float(1 - np.max(new_pattern))}
    return measures