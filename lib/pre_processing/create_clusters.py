import logging
import os
import numpy as np
import sklearn.cluster as cluster

cwd = os.getcwd()
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
        logger.info('Clustering')

        if os.path.isfile(self.cluster_file):
            logger.info('checking if {} exists'.format(self.cluster_file))
            c_centers = np.load(self.cluster_file)
            logger.info('Loaded {} from {} successfully'.format("c_centers", self.cluster_file))
        else:
            logger.info('Performing clustering')
            params = {'n_clusters': num_k,
                      'batch_size': 300,
                      'init': 'k-means++',
                      'random_state': 21
                      }
            k_means = cluster.MiniBatchKMeans(**params)
            k_means.fit(self.doc_vecs)
            c_centers = np.array(k_means.labels_.tolist())
            logger.info('Saving the c_centers data to disk')
            np.save(file=self.cluster_file, arr=c_centers)
        return c_centers


if __name__ == '__main__':
    from lib.pre_processing import CorpusModel, SemanticModels, SimilarityModel
    corpus_model = CorpusModel()
    corpus = corpus_model.load_corpus()
    dictionary = corpus_model.load_dict()
    tfidf_corpus = corpus_model.load_tfidf_corpus()
    semantic_model = SemanticModels(corpus=corpus, dictionary=dictionary, tfidf=False)
    bow_lda = semantic_model.load_lda_model()
    sims_model = SimilarityModel(corpus=bow_lda, dictionary=dictionary, tfidf=False, model='lda')
    sim_index = sims_model.load_sim_index()
    cluster_model = Clusterer(sim_index, tfidf=False, model='lda')
    cluster_model.k_clusterer()
