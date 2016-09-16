import logging
import os
import numpy as np
from sklearn import cluster


logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class Clusterer(object):
    def __init__(self, tmp_folder, corpus, n_docs):
        self.tmp_folder = tmp_folder
        self.doc_vecs = corpus
        self.n_docs = n_docs

    def mini_k_clusters(self, k=14):
        cluster_file = os.path.join(self.tmp_folder, 'k_clusters')
        if os.path.isfile(cluster_file):
            k_labels = np.memmap(cluster_file,
                                 dtype='float32',
                                 mode='r+',
                                 shape=(self.n_docs,)
                                 )
        else:
            params = {'n_clusters': k,
                      'max_iter': 150,
                      'max_no_improvement': 25,
                      'batch_size': 10000,
                      'init': 'k-means++',
                      'n_init': k,
                      'random_state': 24,
                      'reassignment_ratio': 0.09
                      }
            logger.info('Initializing K-Means Clustering')
            k_means = cluster.MiniBatchKMeans(**params)
            k_means.fit(self.doc_vecs)
            logger.info('Storing the cluster labels')
            k_labels = np.memmap(cluster_file,
                                 dtype='float32',
                                 mode='w+',
                                 shape=(self.n_docs,)
                                 )
            k_labels[:] = k_means.labels_
        return k_labels

    def dbscan_clusters(self, dist=0.1, n_samples=10):
        cluster_file = os.path.join(self.tmp_folder, 'k_clusters')
        if os.path.isfile(cluster_file):
            db_labels = np.memmap(cluster_file,
                                  dtype='float32',
                                  mode='r+',
                                  shape=(self.n_docs,)
                                  )
        else:
            params = {'eps': dist,
                      'min_samples': n_samples,
                      'metric': 'precomputed',
                      'n_jobs': -1
                      }
            logger.info('Initializing DBSCAN Clustering')
            dbscan = cluster.DBSCAN(**params)
            dbscan.fit(self.doc_vecs)
            logger.info('storing the cluster labels')
            db_labels = np.memmap(cluster_file,
                                  dtype='float32',
                                  mode='w+',
                                  shape=(self.n_docs,)
                                  )
            db_labels[:] = dbscan.labels_
        return db_labels
