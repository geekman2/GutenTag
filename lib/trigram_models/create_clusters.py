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
import time
import datetime


cwd = settings.project_root
working_directory = os.path.join(cwd, 'tmp/modeldir/')
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG, filename=os.path.join(cwd, 'Clustering.log'))


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

        def compute_time(item_time, remaining_items, i):
            per_item_time = int(5*round(float(item_time)/5))
            if per_item_time < item_time:
                per_item_time += 5
            time_remaining = per_item_time*(remaining_items-i)
            fancy_time = str(datetime.timedelta(seconds=time_remaining)).split(':')
            hours = fancy_time[0]
            minutes = fancy_time[1]
            seconds = fancy_time[2]
            if time_remaining > 3600:
                output = 'Approximately {} hour {} minutes remaining'.format(hours, minutes, seconds)
            elif time_remaining > 60:
                output = 'Approximately {} minutes remaining'.format(minutes)
            else:
                output = 'Approximately {} seconds remaining'.format(seconds)
            return output


        logger.info('Initializing K_Means Clustering')

        if os.path.isfile(self.cluster_file):
            logger.info('checking if {} exists'.format(self.cluster_file))
            cluster_centers = np.load(self.cluster_file)
            logger.info('Loaded {} from {} successfully'.format("cluster_centers", self.cluster_file))
        else:
            logger.info('Performing clustering')
            params = {'n_clusters': 500,
                      'init': 'k-means++',
                      'random_state': 1791,
                      'batch_size': 5000
                      }
            # Initialization
            k_means = cluster.MiniBatchKMeans(**params)
            c_size = 5000
            chunks = gensim.utils.chunkize(corpus, chunksize=c_size, maxsize=3)
            # Training
            for i, item in enumerate(chunks):
                start = time.time()
                npchunk = gensim.matutils.corpus2dense(corpus=item, num_terms=len(dictionary), num_docs=len(item))
                if npchunk.shape[1] == c_size:
                    k_means.partial_fit(npchunk.T)
                    end = time.time() - start
                    remaining_batches = len(corpus)/c_size
                    percentage = round(int(i+1)/float(len(corpus)/c_size))
                    ETA = compute_time(end, remaining_batches, i)
                    log_text = 'Fit batch #{} of {} {}% complete {}'.format(i+1, remaining_batches, percentage, ETA)
                    logger.info(log_text)
                    print log_text

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
    fit_model = cluster_model.k_clusterer()
    cPickle.dump(fit_model, open(os.path.join(cwd, 'KMeans.cluster'), 'w'))
