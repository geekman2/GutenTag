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


class TfidfModel(object):
    def __init__(self, corpus, dictionary=None):
        self.corpus = corpus
        self.dictionary = dictionary
        self.tfidf_file = os.path.join(working_directory,'tfidfCorpora.mm')

    def load_tfidf_corpus(self):
        """
        Load TFIDF model from disk, if it does not exist, train and save it to disk
        :return: TFIDF Corpus
        """
        if os.path.isfile(self.tfidf_file):
            logger.info('trying to load tfidf model from disk')
            self.tfidf_model = gensim.models.tfidfmodel.TfidfModel.load(self.tfidf_file)
            logger.info('loaded tfidf model from disk')
        else:
            logger.info('training tfidf model')
            params = {'corpus': self.corpus,
                      'id2word': self.dictionary
                      }
            self.tfidf_model = gensim.models.tfidfmodel.TfidfModel(**params)
            logger.info('Saving tfidf model to disk')
            self.tfidf_model.save(self.tfidf_file)
            logger.info('Saved tfidf model to disk')
        self.tfidf_corpus = self.tfidf_model[self.corpus]
        return self.tfidf_corpus

if __name__ == '__main__':
    from lib.pre_processing import CorpusModel
    corpus_model = CorpusModel()
    corpus = corpus_model.load_corpus()
    dictionary = corpus_model.load_dict()
    tfidf_model = TfidfModel(corpus, dictionary)
    tfidf_corpus = tfidf_model.load_tfidf_corpus()
