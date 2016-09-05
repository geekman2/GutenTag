import gensim
import logging
import os
import time
import itertools

import lib.baselineModels.textCleaner as cleaner
import lib.WordVectors.parser as mongoClient

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# Global configurations
cwd = os.getcwd()
working_directory = os.path.join(cwd,'tmp','modeldir')
if not os.path.exists(working_directory):
    os.makedirs(working_directory)
docs = mongoClient.docs
processed_docs = cleaner.cleanText(docs=docs)

class MyCorpus(gensim.corpora.TextCorpus):

    def get_texts(self):
        for doc in self.input:
            yield doc

    def get_docs(self):
        for item in self.get_texts():
            yield self.dictionary.doc2bow(item, allow_update=False)

    def load_dict(self):
        """

        :return:
        """
        self.dict_file = os.path.join(working_directory, 'corpus.dict')
        if os.path.isfile(self.dict_file):
            logger.info('Loading corpus dictionary from file')
            self.dictionary = gensim.corpora.dictionary.Dictionary.load(self.dict_file)
            logger.info('Corpus dictionary loaded from file')
        else:
            # self.docs, self.idx = itertools.izip(*self.get_text())
            self.dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
            self.dictionary.compactify()
            self.dictionary.save(self.dict_file)
        return self.dictionary

    def load_corpus(self):
        """
        Load corpus if it exists, else build corpus from docs
        Corpus being defined as the vector bag of words representation of all documents
        :rtype: object
        :return: Corpus
        """
        self.corpus_file = os.path.join(working_directory, 'corpora.mm')
        if os.path.isfile(self.corpus_file):
            logger.info('Trying to load corpus from disk')
            self.corpus = gensim.corpora.mmcorpus.MmCorpus(self.corpus_file)
            logger.info('loaded corpus from disk')
        else:
            logger.info('Corpus not found on disk. Building corpus and serializing to disk')
            gensim.corpora.mmcorpus.MmCorpus.serialize(self.corpus_file, self.get_docs())
            logger.info('Successfully built corpus and saved corpus to disk')
            self.corpus = gensim.corpora.mmcorpus.MmCorpus(self.corpus_file)
        return self.corpus

    def load_tfidf_corpus(self):
        """
        Load TFIDF model from disk, if it does not exist, train and save it to disk
        :return: TFIDF Corpus
        """
        self.tfidf_file = os.path.join(working_directory, 'tfidfCorpora.mm')
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
    # dataFile = '{}/tmp/genData.json'.format(os.getcwd())
    # cursy = simMongoDb(n=10, array=True, jsonLoc=dataFile)
    start = time.time()

    the_model = MyCorpus(processed_docs)
    the_model.load_dict()
    the_model.load_corpus()
    the_model.load_tfidf_corpus()
