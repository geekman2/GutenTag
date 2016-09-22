import os
import numpy as np
from sklearn import cluster
import gensim

from time import time
import logging

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class Clusterer(object):
    def __init__(self, tmp_folder, corpus, n_docs):
        self.tmp_folder = tmp_folder
        self.doc_vecs = corpus
        self.n_docs = n_docs

    def chunkize_corpus(self, c_size=5000):
        for chunk in gensim.matutils.utils.chunkize(self.doc_vecs,
                                                    chunksize=c_size):
            yield chunk

    def numpize(self, chunk):
        return gensim.matutils.corpus2dense(corpus=chunk,
                                            num_docs=len(chunk),
                                            num_terms=corpus.num_terms)

    def mini_k_clusters(self, k=5, corpus_type=None):
        cluster_file = os.path.join(self.tmp_folder, '{}_{}_clusters'.format(corpus_type, k))
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
                      'batch_size': 5000,
                      'init': 'k-means++',
                      'n_init': k,
                      'random_state': 24,
                      'reassignment_ratio': 0.09
                      }
            logger.info('Initializing K-Means Clustering')
            k_means = cluster.MiniBatchKMeans(**params)
            i = 1
            for chunk in self.chunkize_corpus():
                np_chunk = self.numpize(chunk=chunk)
                logger.info('Fitting chunk {}'.format(i))
                k_means.partial_fit(np_chunk.T)
                i += 1
            logger.info('Storing the cluster labels')
            k_labels = np.memmap(cluster_file,
                                 dtype='float32',
                                 mode='w+',
                                 shape=(self.n_docs,)
                                 )
            i = 0
            j = 1
            for chunk in self.chunkize_corpus():
                np_chunk = self.numpize(chunk=chunk)
                logger.info('predicting chunk {}'.format(j))
                labels = k_means.predict(np_chunk.T)
                k_labels[i:i+labels.shape[0]] = labels
                i += labels.shape[0]
                j += 1
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


if __name__ == '__main__':
    from lib.topic_models.vector_models import VectorModels
    from lib.topic_models.semantic_models import TopicModels

    start_corpus = time()
    cwd = os.getcwd()
    data_loc = os.path.join(cwd, 'tmp', 'text_corpus.dat', )
    tmp_folder = os.path.join(cwd, 'tmp', 'modeldir')

    vectors = VectorModels(data_loc, tmp_folder)
    corpus, dictionary = vectors.load_corpus()
    tfidf_corpus = vectors.build_tfidf_corpus(corpus, dictionary)

    semantic = TopicModels(tmp_folder, tfidf_corpus, dictionary)
    lda_corpus = semantic.build_lda_corpus(bow=False)

    clustering = Clusterer(tmp_folder,tfidf_corpus, n_docs=tfidf_corpus.num_docs)
    k_labels = clustering.mini_k_clusters(k=5, corpus_type='tfidf')