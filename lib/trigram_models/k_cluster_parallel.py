#  -------------------------------------------------------------------------------
# Name:         K Cluster Parallel
# Purpose:      Compute KMeans on corpus in parallel in order to finish before the heat
#               death of the universe
# Author:       Devon Muraoka, Bharat Ramanthan
# Created:      9/16/16
# Copyright:   (c) Devon Muraoka, Bharat Ramanathan 
#  -------------------------------------------------------------------------------
from __future__ import print_function
import logging
import os
import numpy as np
import sklearn.cluster as cluster
import settings
import cPickle
import gensim
import multiprocessing

cwd = settings.project_root
working_directory = os.path.join(cwd, 'tmp/modeldir/')
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)

def cluster_chunk(pair):
    i,item = pair
    npchunk = gensim.matutils.corpus2dense(corpus=item, num_terms=len(dictionary), num_docs=len(item))
    print('stuff')
    if npchunk.shape[1] == c_size:
        k_means.partial_fit(npchunk.T)
        logger.info('Fit batch #{}'.format(i + 1))
        print('Fit batch #{}'.format(i + 1))


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
    tfidf = True
    model = None
    doc_vecs = tfidf_corpus

    if tfidf and model == 'lda':
        cluster_file = os.path.join(working_directory, 'tfidf_lda_cluster.npy')
    elif tfidf and model == 'lsa':
        cluster_file = os.path.join(working_directory, 'tfidf_lsa_cluster.npy')
    elif not tfidf and model == 'lda':
        cluster_file = os.path.join(working_directory, 'bow_lda_cluster.npy')
    elif not tfidf and model == 'lsa':
        cluster_file = os.path.join(working_directory, 'bow_lsa_cluster.npy')
    elif tfidf and not model:
        cluster_file = os.path.join(working_directory, 'tfidf_cluster.npy')
    else:
        cluster_file = os.path.join(working_directory, 'kmeans_tfidf_cluster.npy')

    logger.info('Initializing K_Means Clustering')

    if os.path.isfile(cluster_file):
        logger.info('checking if {} exists'.format(cluster_file))
        cluster_centers = np.load(cluster_file)
        logger.info('Loaded {} from {} successfully'.format("cluster_centers", cluster_file))
    else:
        logger.info('Performing clustering')
        params = {'n_clusters': 21,
                  'init': 'k-means++',
                  'random_state': 1791,
                  'batch_size': 1000
                  }
        # Initialization
        k_means = cluster.MiniBatchKMeans(**params)
        c_size = 1000
        chunks = gensim.utils.chunkize(corpus, chunksize=c_size, maxsize=3)
        # Training
        p = multiprocessing.Pool(8)
        pack = enumerate(chunks)
        p.map_async(cluster_chunk,pack)
        p.close()
        p.join()
