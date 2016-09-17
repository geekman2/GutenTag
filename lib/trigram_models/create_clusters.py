# coding = utf-8
# ------------------------------------------------------------------------------
# Name:         Create clusters
# Purpose:      Create clusters#TODO improve this description
# Author:       Bharat Ramanathan, Devon Muraoka
# Created:      9/6/2016
# Copyright:    (c) Bharat Ramanathan, Devon Muraoka
# ------------------------------------------------------------------------------
import logging
import os
import numpy as np
import sklearn.cluster as cluster
import settings
import cPickle
import gensim


cwd = settings.project_root
working_directory = os.path.join(cwd, 'tmp/modeldir/')
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)


class Clusterer(object):
    def __init__(self, doc_vecs, tfidf=True, model=None):
        self.doc_vecs = doc_vecs
        if tfidf and model == 'lda':
            self.cluster_file = os.path.join(working_directory, 'tfidf_lda_cluster.npy')
        elif tfidf and model == 'lsa':
            self.cluster_file = os.path.join(working_directory, 'tfidf_lsa_cluster.npy')
        elif not tfidf and model == 'lda':
            self.cluster_file = os.path.join(working_directory, 'bow_lda_cluster.npy')
        elif not tfidf and model == 'lsa':
            self.cluster_file = os.path.join(working_directory, 'bow_lsa_cluster.npy')
        elif tfidf and not model:
            self.cluster_file = os.path.join(working_directory, 'tfidf_cluster.npy')
        else:
            self.cluster_file = os.path.join(working_directory, 'kmeans_tfidf_cluster.npy')

    def k_clusterer(self, num_k=21):

        logger.info('Initializing K_Means Clustering')

        if os.path.isfile(self.cluster_file):
            logger.info('checking if {} exists'.format(self.cluster_file))
            cluster_centers = np.load(self.cluster_file)
            logger.info('Loaded {} from {} successfully'.format("cluster_centers", self.cluster_file))
        else:
            logger.info('Performing clustering')
            params = {'n_clusters': 21,
                      'init': 'k-means++',
                      'random_state': 1791,
                      'batch_size':1000
                      }
            # Initialization
            k_means = cluster.MiniBatchKMeans(**params)
            c_size = 1000
            chunks = gensim.utils.chunkize(corpus, chunksize=c_size, maxsize=3)
            # Training
            for i,item in enumerate(chunks):
                npchunk = gensim.matutils.corpus2dense(corpus=item, num_terms=len(dictionary), num_docs=len(item))
                if npchunk.shape[1] == c_size:
                    k_means.partial_fit(npchunk.T)
                    logger.info('Fit batch #{}'.format(i+1))
            #Parallel(n_jobs=-1)(delayed(worker)(i) for i in chunks)
        return k_means


    def agglo_clusterer(self, num_k=21):
        logger.info('Initializing Agglomerative Clustering')
        params = {'n_clusters': num_k,
                  'affinity ': 'precomputed',
                  }
        agglo = cluster.AgglomerativeClustering(**params)
        agglo.fit(self.doc_vecs)
        cluster_centers = np.array(agglo.labels_.tolist())
        return cluster_centers

    def dbscan_clusterer(self):
        logger.info('Performing Clustering using DBSCAN')
        params = {'metric': 'precomputed'
                  }
        dbscan = cluster.DBSCAN(**params)
        dbscan.fit(self.doc_vecs)
        cluster_centers = np.array(dbscan.labels_.tolist())
        return cluster_centers


if __name__ == '__main__':
    from lib.trigram_models import CorpusModel, SemanticModels, SimilarityModel

    logger.info(working_directory)
    corpus_model = CorpusModel()
    corpus = corpus_model.load_corpus()
    dictionary = corpus_model.load_dict()
    tfidf_corpus = corpus_model.load_tfidf_corpus()
    # semantic_model = SemanticModels(corpus=corpus, dictionary=dictionary, tfidf=False)
    # bow_lda = semantic_model.load_lda_model()
    # sims_model = SimilarityModel(corpus=tfidf_corpus, dictionary=dictionary, tfidf=True, model=None)
    # sim_index = sims_model.load_sim_index()
    # sim_index.output_prefix = working_directory
    # sim_index.check_moved()
    cluster_model = Clusterer(tfidf_corpus, tfidf=True, model=None)
    fit_model = cluster_model.k_clusterer()[0]
    cPickle.dump(fit_model, open('KMeans.cluster', 'w'))
