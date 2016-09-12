import os
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np


class DistanceMetrics(object):
    def _jensen_shannon(self, u, v):
        _u = u / norm(u, ord=1)
        _v = v / norm(v, ord=1)
        _M = 0.5 * (_u + _v)
        return 0.5 * (entropy(_u, _M) + entropy(_v, _M))

    def build_similarity_index(self, corpus, distance_metric='JS'):
        if distance_metric == 'JS':
            sim_index_file = os.path.join()
            pairwise_distances(corpus)
