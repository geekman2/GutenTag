# coding = utf-8
# ------------------------------------------------------------------------------
# Name:         Create corpus models
# Purpose:      Create corpus models#TODO improve this description
# Author:       Bharat Ramanathan, Devon Muraoka
# Created:      9/6/2016
# Copyright:    (c) Bharat Ramanathan, Devon Muraoka
# ------------------------------------------------------------------------------
from __future__ import print_function, absolute_import
import var.settings as settings
import os

import logging
import gensim


logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

cwd = os.getcwd()
working_directory = os.path.join(cwd,'tmp','modeldir')
if not os.path.exists(working_directory):
    os.makedirs(working_directory)


class CorpusModel(object):
    def get_texts(self):
        self.cur = settings.docs.find({'text': {'$exists': 1}}, {'text': 1, '_id': 0}).batch_size(10000)
        for doc in self.cur:
            yield doc['text']

    def get_trigrams(self):
        for text in self.get_texts():
            tokens = []
            for i in xrange(len(text) - 3):
                tokens.append(text[i:i+3])
            yield tokens

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
            self.dictionary = gensim.corpora.Dictionary(self.get_trigrams())
            self.dictionary.filter_extremes(no_below=1, no_above=0.99, keep_n=100000)
            self.dictionary.compactify()
            self.dictionary.save(self.dict_file)
        return self.dictionary

    def get_bow(self):
        for doc in self.get_trigrams():
            yield self.dictionary.doc2bow(doc)

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
            gensim.corpora.mmcorpus.MmCorpus.serialize(self.corpus_file, self.get_bow(), id2word=self.dictionary)
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
    mycorpus = CorpusModel()
    mycorpus.load_dict()
    mycorpus.load_corpus()
    mycorpus.load_tfidf_corpus()