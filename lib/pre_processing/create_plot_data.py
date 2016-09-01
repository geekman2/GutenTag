import logging
import os
import numpy as np
import sklearn.manifold as manifold

cwd = os.getcwd()
working_directory = '{}/tmp/modeldir/'.format(cwd)
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class ReduceDimension():
    def __init__(self, data, tfidf=True, model=None):
        self.data = data
        if tfidf and model == 'lda':
            self.t_sne_file = '{}tfidf_lda_t_sne.npy'.format(working_directory)
        elif tfidf and model == 'lsa':
            self.t_sne_file = '{}tfidf_lsa_t_sne.npy'.format(working_directory)
        elif not tfidf and model == 'lda':
            self.t_sne_file = '{}bow_lda_t_sne.npy'.format(working_directory)
        elif not tfidf and model == 'lsa':
            self.t_sne_file = '{}bow_lsa_t_sne.npy'.format(working_directory)
        elif tfidf and not model:
            self.t_sne_file = '{}tfidf_t_sne.npy'.format(working_directory)

    def t_sne_reduce(self):
        if os.path.isfile(self.t_sne_file):
            logger.info('Checking if {} exists'.format(self.t_sne_file))
            t_sne_data = np.load(self.t_sne_file)
        else:
            logger.info('Reducing Dimensionality')
            t_sne = manifold.TSNE()
            t_sne_data = t_sne.fit_transform(self.data)
            logger.info('Saving the t_sne_data to disk')
            np.save(file=self.t_sne_file, arr=t_sne_data)
        return t_sne_data


if __name__ == '__main__':
    from lib.pre_processing import CorpusModel, TfidfModel, SemanticModels, SimilarityModel
    corpus = CorpusModel().load_corpus()
    dictionary = CorpusModel().load_dict()
    tfidf_model = TfidfModel(corpus, dictionary)
    tfidf_corpus = tfidf_model.load_tfidf_corpus()
    semantic_model = SemanticModels(corpus=tfidf_corpus, dictionary=dictionary, tfidf=True)
    tfidf_lda = semantic_model.load_lda_model()
    sims_model = SimilarityModel(corpus=tfidf_lda, dictionary=dictionary, tfidf=True, model='lda')
    sim_index = sims_model.load_sim_index()
    reducer = ReduceDimension(sim_index, tfidf=True, model='lda')
    reducer.t_sne_reduce()
