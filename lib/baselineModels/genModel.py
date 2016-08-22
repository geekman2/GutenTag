from __future__ import print_function, absolute_import
from var.mongoSim import simMongoDb
# import numpy as np
import itertools
import os
from gensim import corpora, utils
from gensim.corpora import dictionary
from gensim.models import tfidfmodel, ldamulticore
from time import time


class textModel(object):
    def __init__(self, cur):
        self.cur = cur
        self.docs, self.ids = itertools.izip(*self.getText())
        self._generateConf()

    def getText(self):
        for item in self.cur:
            yield item['text'].split(), item['_id']

    def _generateConf(self):
        cwd = os.getcwd()
        self.dictFile = "{}/tmp/corpus.dict".format(cwd)
        self.corpusFile = "{}/tmp/corpora.mm".format(cwd)
        self.tfidfFile = "{}/tmp/tfidfCorpora.mm".format(cwd)


class corpusModel(textModel):
    def __init__(self, cur):
        textModel.__init__(self, cur)
        self.loadDict()
        self.dictionary.filter_extremes(no_below=2, no_above=0.5,
                                        keep_n=100000)
        self.dictionary.compactify()
        self.corpus = self.loadCorpus()
        self.tfidfCorpus = self.loadTfidfCorpus()

    """
    def __iter__(self):
        for key, value in self.dictionary.iteritems():
            yield (key, value)
    """

    def buildDoc2Bow(self):
        for doc in self.docs:
            yield self.dictionary.doc2bow(doc)

    def loadDict(self):
        try:
            self.dictionary = dictionary.Dictionary.load(self.dictFile)
        except IOError:
            self.dictionary = dictionary.Dictionary(self.docs)
            self.dictionary.save(self.dictFile)

    def loadCorpus(self):
        try:
            return corpora.mmcorpus.MmCorpus(self.corpusFile)
        except IOError:
            corpus = self.buildDoc2Bow()
            corpora.mmcorpus.MmCorpus.serialize(self.corpusFile, corpus)
            return corpus

    def loadTfidfCorpus(self):
        try:
            tfidf = tfidfmodel.TfidfModel.load(self.tfidfFile)
            tfidfCorpus = tfidf[self.corpus]
            return tfidfCorpus
        except IOError:
            tfidf = tfidfmodel.TfidfModel(self.corpus,
                                          id2word=self.dictionary)
            tfidfCorpus = tfidf[self.corpus]
            tfidf.save(self.tfidfFile)
            return tfidfCorpus

    def loadLDAModel(self):
        lda = ldamulticore.LdaMulticore(corpus=self.tfidfCorpus,
                                        id2word=self.dictionary, workers=16)
        print(lda.print_topics(10))


def mapper(dictModel):
    print(utils.revdict(dictModel.getToken())[1273])

if __name__ == '__main__':
    dataFile = "{}/tmp/bowdata.json".format(os.getcwd())
    cur = simMongoDb(n=10000, array=True, jsonLoc=dataFile)
    start = time()
    theModel = corpusModel(cur)
    # print(theModel.tfidfCorpus)
    theModel.loadLDAModel()
    # print(time() - start)
