import os

import gensim

from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np

from time import time
import logging


logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def jensen_shannon(u, v):
    _u = u / norm(u, ord=1)
    _v = v / norm(v, ord=1)
    _M = 0.5 * (_u + _v)
    d = 0.5 * (entropy(_u, _M) + entropy(_v, _M))
    if np.isinf(d):
        d = 0
    return d


def kullback_leibler(u, v):
    d = gensim.matutils.kullback_leibler(u, v)
    if np.isinf(d):
        d = 0
    return d


class DistanceMetrics(object):
    def __init__(self, tmp_dir, corpus):
        self.tmp_dir = tmp_dir
        self.n_docs = corpus.num_docs
        self.n_terms = corpus.num_terms
        self.corpus = self._get_corpus_mmap(corpus)

    def _read_mmap(self, map_file, shape, dtype='float32'):
        return np.memmap(map_file, dtype=dtype, shape=shape, mode='r+')

    def _write_mmap(self, map_file, shape, dtype='float32'):
        return np.memmap(map_file, dtype=dtype, shape=shape, mode='w+')

    def _get_corpus_mmap(self, corpus):
        corpus_map_file = os.path.join(self.tmp_dir, 'corpus_map.mmep')
        if os.path.isfile(corpus_map_file):
            corpus_map = self._read_mmap(map_file=corpus_map_file,
                                         shape=(self.n_terms, self.n_docs)
                                         )
        else:
            corpus_map = self._write_mmap(map_file=corpus_map_file,
                                          shape=(self.n_terms, self.n_docs)
                                          )
            params = {'corpus': corpus,
                      'num_terms': self.n_terms,
                      'num_docs': self.n_docs
                      }
            corpus_map[:] = gensim.matutils.corpus2dense(**params)
        return corpus_map.T

    def _jensen_shannon(u, v):
        _u = u / norm(u, ord=1)
        _v = v / norm(v, ord=1)
        _M = 0.5 * (_u + _v)
        return 0.5 * (entropy(_u, _M) + entropy(_v, _M))

    def _get_metric_config(self, metric, fout=False):
        default_dict = {'X': self.corpus,
                        'n_jobs': -1
                        }
        config_dict = {'jensen_shannon': jensen_shannon,
                       'cosine': 'cosine',
                       'jaccard': 'jaccard',
                       'kullback_leibler': kullback_leibler,
                       'hellinger': gensim.matutils.hellinger,
                       }
        default_dict['metric'] = config_dict[metric]
        if fout:
            default_dict['fname'] = metric+'.idx'
        return default_dict

    def build_similarity_index(self, distance_metric='jensen_shannon'):
        params = self._get_metric_config(distance_metric)
        return pairwise_distances(**params)

    def load_similarity_index(self, metric='jensen_shannon'):
        file_path = self._get_metric_config(metric, fout=True)['fname']
        sim_index_file = os.path.join(self.tmp_dir, file_path)
        if os.path.isfile(sim_index_file):
            sim_index = self._read_mmap(sim_index_file,
                                        shape=(self.n_docs, self.n_docs)
                                        )
        else:
            sim_index = self._write_mmap(sim_index_file,
                                         shape=(self.n_docs, self.n_docs)
                                         )
            sim_index[:] = self.build_similarity_index(metric)
        return sim_index


if __name__ == '__main__':
    from lib.topicModels.vector_models import VectorModels
    from lib.topicModels.semantic_models import TopicModels

    start_corpus = time()
    cwd = os.getcwd()
    data_folder = os.path.join(cwd, 'tmp', 'testFiles', '*')
    tmp_folder = os.path.join(cwd, 'tmp', 'modeldir')

    vectors = VectorModels(data_folder, tmp_folder)
    corpus, dictionary = vectors.load_corpus()
    tfidf_corpus = vectors.build_tfidf_corpus(corpus, dictionary)

    semantic = TopicModels(tmp_folder, tfidf_corpus, dictionary)
    lda_corpus = semantic.build_lda_corpus()
    hdp_corpus = semantic.build_hdp_corpus()

    lda_metrics = DistanceMetrics(tmp_folder, lda_corpus)

    lda_hellidx = lda_metrics.load_similarity_index(metric='hellinger')
    lda_jensen_shannonidx = lda_metrics.load_similarity_index(
        metric='jensen_shannon')
    lda_kullback_leibleridx = lda_metrics.load_similarity_index(
        metric='kullback_leibler')

    hdp_metrics = DistanceMetrics(tmp_folder, hdp_corpus)
    hdp_hellidx = hdp_metrics.load_similarity_index(metric='hellinger')
    hdp_jensen_shannonidx = hdp_metrics.load_similarity_index(
        metric='jensen_shannon')
    hdp_kullback_leibleridx = hdp_metrics.load_similarity_index(
        metric='kullback_leibler')

    # print(metrics.corpus.sum(axis=1))

    stop_corpus = time()
    corpus_time = round(stop_corpus - start_corpus, 3)
    logger.info('time taken to build corpus = {}s'.format(corpus_time))
