# coding = utf-8
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


class DistanceMetrics(object):
    def __init__(self, tmp_dir, corpus):
        self.tmp_dir = tmp_dir
        self.n_docs = corpus.num_docs
        self.n_terms = corpus.num_terms
        self.n_nnz = corpus.num_nnz
        self.corpus = corpus

    def build_cosine_similarity_index(self, type='bow'):
        output_prefix = os.path.join(self.tmp_dir, '{}_cosine_shard'.format(type))
        sim_index_file = os.path.join(self.tmp_dir, '{}_cosine_index'.format(type))
        if os.path.isfile(sim_index_file):
            sim_index = gensim.similarities.Similarity.load(sim_index_file)
        else:
            params = {'output_prefix': output_prefix,
                      'corpus': self.corpus,
                      'num_features': self.n_terms,
                      'num_best': self.n_docs
                      }
            sim_index = gensim.similarities.Similarity(**params)
            sim_index.save(sim_index_file)
        return sim_index

    def get_sparse_matrix(self):
        sparse_corpus = gensim.matutils.corpus2csc(corpus=self.corpus,
                                                   num_terms=self.n_terms,
                                                   num_docs=self.n_docs,
                                                   num_nnz=self.n_nnz
                                                   )
        return sparse_corpus

    def build_jaccard_sim(self):
        mat = self.get_sparse_matrix()
        cols_sum = mat.getnnz(axis=0)
        ab = mat.T * mat

        # for rows
        aa = np.repeat(cols_sum, ab.getnnz(axis=0))
        # for columns
        bb = cols_sum[ab.indices]

        similarities = ab.copy()
        similarities.data /= (aa + bb - ab.data)

        return similarities


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

    metrics = DistanceMetrics(tmp_dir=tmp_folder, corpus=lda_corpus)
    sims = metrics.build_cosine_similarity_index(type='tdidf_lda')

    sims = gensim.matutils.corpus2csc(sims,num_terms=100000, dtype=np.float32, num_docs=100000, printprogress=1)
    print(sims)
    stop_corpus = time()
    corpus_time = round(stop_corpus - start_corpus, 3)
    logger.info('time taken to build corpus = {}s'.format(corpus_time))
