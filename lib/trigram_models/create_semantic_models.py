import os
import logging
import gensim


logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# Global configure Files
cwd = os.getcwd()
working_directory = os.path.join(cwd,'tmp','modeldir')
if not os.path.exists(working_directory):
    os.makedirs(working_directory)


class SemanticModels(object):
    def __init__(self, corpus, dictionary=None, tfidf=True):
        self.corpus = corpus
        self.dictionary = dictionary
        if tfidf:
            self.lda_file = os.path.join(working_directory,'tfidf_ldaModel.mm')
            self.lsa_file = os.path.join(working_directory,'tfidf_lsaModel.mm')
        else:
            self.lda_file = os.path.join(working_directory,'bow_ldaModel.mm')
            self.lsa_file = os.path.join(working_directory,'bow_lsaModel.mm')

    def load_lda_model(self, n_topics=21):
        """
        Load Latent Dirichlet Allocation Model into memory, if none exists, create it
        :param n_topics: The number of topics the LDA algorithm will pick out
        :return:
        """
        if os.path.isfile(self.lda_file):
            logger.info('trying to load LDA model from disk')
            self.lda_model = \
                gensim.models.ldamulticore.LdaMulticore.load(self.lda_file)
            logger.info('loaded LDA model from disk')
        else:
            logger.info('training LDA model')
            params = {'corpus': self.corpus,
                      'id2word': self.dictionary,
                      'num_topics': n_topics,
                      'workers': 4
                      }
            self.lda_model = gensim.models.ldamulticore.LdaMulticore(**params)
            logger.info('Saving LDA model to disk')
            self.lda_model.save(self.lda_file)
            logger.info('Saved LDA model to disk')
        self.lda_corpus = self.lda_model[self.corpus]
        return self.lda_corpus

    def load_lsa_model(self, n_topics=21):
        """
        Load Latent Semantic Analysis Model into memory, if none exists, create it
        :param n_topics: The number of topics the LSA algorithm will pick out
        :return:
        """
        if os.path.isfile(self.lsa_file):
            logger.info('trying to load LDA model from disk')
            self.lsa_model = gensim.models.lsimodel.LsiModel.load(self.lsa_file)
            logger.info('loaded LSA model from disk')
        else:
            logger.info('training LSA model')
            params = {'corpus': self.corpus,
                      'id2word': self.dictionary,
                      'num_topics': n_topics,
                      }
            self.lsa_model = gensim.models.lsimodel.LsiModel(**params)
            logger.info('Saving LSA model to disk')
            self.lsa_model.save(self.lsa_file)
            logger.info('Saved LSA model to disk')
        self.lsa_corpus = self.lsa_model[self.corpus]
        return self.lsa_corpus

if __name__ == '__main__':
    from lib.trigram_models import CorpusModel
    corpus_model = CorpusModel()
    corpus = corpus_model.load_corpus()
    dictionary = corpus_model.load_dict()
    tfidf_corpus = corpus_model.load_tfidf_corpus()
    semantic_model = SemanticModels(corpus=corpus, dictionary=dictionary, tfidf=False)
    semantic_model.load_lda_model()
