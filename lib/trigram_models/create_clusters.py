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

cwd = settings.project_root
working_directory = os.path.join(cwd,'tmp','modeldir')
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class Clusterer(object):
    def __init__(self, doc_vecs, tfidf=True, model=None):
        self.doc_vecs = doc_vecs
        if tfidf and model == 'lda':
            self.cluster_file = os.path.join(working_directory,'tfidf_lda_cluster.npy')
        elif tfidf and model == 'lsa':
            self.cluster_file = os.path.join(working_directory,'tfidf_lsa_cluster.npy')
        elif not tfidf and model == 'lda':
            self.cluster_file = os.path.join(working_directory,'bow_lda_cluster.npy')
        elif not tfidf and model == 'lsa':
            self.cluster_file = os.path.join(working_directory,'bow_lsa_cluster.npy')
        elif tfidf and not model:
            self.cluster_file = os.path.join(working_directory,'tfidf_cluster.npy')

    def k_clusterer(self, num_k=21):
        logger.info('Initializing K_Means Clustering')

        if os.path.isfile(self.cluster_file):
            logger.info('checking if {} exists'.format(self.cluster_file))
            cluster_centers = np.load(self.cluster_file)
            logger.info('Loaded {} from {} successfully'.format("cluster_centers", self.cluster_file))
        else:
            logger.info('Performing clustering')
            params = {'n_clusters': num_k,
                      'batch_size': 300,
                      'init': 'k-means++',
                      'random_state': 21
                      }
            k_means = cluster.MiniBatchKMeans(**params)
            k_means.fit(self.doc_vecs)
            cluster_centers = np.array(k_means.labels_.tolist())
            logger.info('Saving the cluster_centers data to disk')
            np.save(file=self.cluster_file, arr=cluster_centers)
        return cluster_centers

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
        params = {'metric':'precomputed'
                    }
        dbscan = cluster.DBSCAN(**params)
        dbscan.fit(self.doc_vecs)
        cluster_centers = np.array(dbscan.labels_.tolist())
        return cluster_centers

if __name__ == '__main__':
    from lib.trigram_models import CorpusModel, SemanticModels, SimilarityModel
    corpus_model = CorpusModel()
    corpus = corpus_model.load_corpus()
    dictionary = corpus_model.load_dict()
    tfidf_corpus = corpus_model.load_tfidf_corpus()
    #semantic_model = SemanticModels(corpus=corpus, dictionary=dictionary, tfidf=False)
    #bow_lda = semantic_model.load_lda_model()
    sims_model = SimilarityModel(corpus=tfidf_corpus, dictionary=dictionary, tfidf=True, model=None)
    sim_index = sims_model.load_sim_index()
    cluster_model = Clusterer(sim_index, tfidf=False, model='lda')
    cluster_model.k_clusterer()
