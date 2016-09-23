
import os
import logging
import gensim
import settings

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# Global configure Files
cwd = settings.project_root
working_directory = os.path.join(cwd,'tmp','modeldir')
if not os.path.exists(working_directory):
    os.makedirs(working_directory)


class SimilarityModel(object):
    def __init__(self, corpus, dictionary=None, tfidf=True, model=None):
        self.corpus = corpus
        self.dictionary = dictionary
        if tfidf and model == 'lda':
            self.sim_index_file = os.path.join(working_directory,'tfidf_lda_simFile.idx')
            self.simIndexPrefix = os.path.join(working_directory,'tfidf_lda_simidx')
        elif tfidf and model == 'lsa':
            self.sim_index_file = os.path.join(working_directory,'tfidf_lsa_simFile.idx')
            self.simIndexPrefix = os.path.join(working_directory,'tfidf_lsa_simidx')
        elif not tfidf and model == 'lda':
            self.sim_index_file = os.path.join(working_directory,'bow_lda_simFile.idx')
            self.simIndexPrefix = os.path.join(working_directory,'bow_lda_simidx')
        elif not tfidf and model == 'lsa':
            self.sim_index_file = os.path.join(working_directory,'bow_lsa_simFile.idx')
            self.simIndexPrefix = os.path.join(working_directory,'bow_lsa_simidx')
        elif tfidf and not model:
            self.sim_index_file = os.path.join(working_directory,'tfidf_simFile.idx')
            self.simIndexPrefix = os.path.join(working_directory,'tfidf_simidx')
        logger.info("INDEX FILE LOCATION:{}".format(self.sim_index_file))

    def load_sim_index(self, n_features=None, n_best=None):
        """
        Return index of similarity
        :param n_features:
        :param n_best: Number of documents to return, sorted by most similar
        :return:Similar Object:
        """
        if not n_features:
            n_features = len(self.dictionary)
        if os.path.isfile(self.sim_index_file):
            logger.info('Loading Sim Index from disk')
            self.similar_index = gensim.similarities.docsim.Similarity.load(self.sim_index_file)
            logger.info('Loaded Sim Index from disk')
        else:
            params = {'output_prefix': self.simIndexPrefix,
                      'corpus': self.corpus,
                      'num_features': n_features,
                      'num_best': n_best
                      }
            logger.info('Creating Sim Index')
            self.similar_index = gensim.similarities.docsim.Similarity(**params)
            logger.info('Saving Sim Index to disk')
            self.similar_index.save(self.sim_index_file)
            logger.info('Saved Sim Index to disk')
        return self.similar_index

if __name__ == '__main__':
    from lib.trigram_models import CorpusModel, SemanticModels
    corpus_model = CorpusModel()
    corpus = corpus_model.load_corpus()
    dictionary = corpus_model.load_dict()
    tfidf_corpus = corpus_model.load_tfidf_corpus()
    # semantic_models = SemanticModels(corpus=corpus, dictionary=dictionary, tfidf=False)
    # bow_lda = semantic_models.load_lda_model()
    sims_model = SimilarityModel(corpus=tfidf_corpus, dictionary=dictionary, tfidf=True, model=None)
    sim_index = sims_model.load_sim_index()
