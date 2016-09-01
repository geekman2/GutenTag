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
working_directory = '{}/tmp/modeldir/'.format(cwd)
if not os.path.exists(working_directory):
    os.makedirs(working_directory)
docs = mongoClient.docs
cur = docs.find({'text': {'$exists': 'true'}}, {'text': 1})
cursor = cleaner.cleanText(cur[:10000])


class CorpusModel(object):
    def __init__(self):
        self.cur = cursor
        self.dict_file = '{}corpus.dict'.format(working_directory)
        self.corpus_file = '{}corpora.mm'.format(working_directory)

    def get_text(self):
        """
        Generate doc/id pairs from self.cur
        """
        for item in self.cur:
            yield item['text'].split(), item['_id']

    def build_doc_2_bow(self):
        """

        :return:
        """
        return [self.dictionary.doc2bow(doc) for doc in self.docs]

    def load_dict(self):
        """

        :return:
        """
        if os.path.isfile(self.dict_file):
            logger.info('Loading corpus from file')
            self.dictionary = gensim.corpora.dictionary.Dictionary.load(self.dict_file)
            logger.info('Corpus loaded from file')
        else:
            self.docs, self.idx = itertools.izip(*self.get_text())
            self.dictionary = gensim.corpora.dictionary.Dictionary(self.docs)
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
        if os.path.isfile(self.corpus_file):
            logger.info('Trying to load corpus from disk')
            self.corpus = gensim.corpora.mmcorpus.MmCorpus(self.corpus_file)
            logger.info('loaded corpus from disk')
        else:
            self.dictionary = self.load_dict()
            logger.info('Corpus not found on disk. Building corpus and serializing to disk')
            gensim.corpora.mmcorpus.MmCorpus.serialize(self.corpus_file, self.build_doc_2_bow())
            logger.info('Successfully built corpus and saved corpus to disk')
            self.corpus = gensim.corpora.mmcorpus.MmCorpus(self.corpus_file)
        return self.corpus

if __name__ == '__main__':
    # dataFile = '{}/tmp/genData.json'.format(os.getcwd())
    # cursy = simMongoDb(n=10, array=True, jsonLoc=dataFile)
    start = time.time()
    the_model = CorpusModel()
    the_model.load_dict()
    the_model.load_corpus()
