from __future__ import print_function, absolute_import
from var.mongoSim import simMongoDb
import numpy as np
import itertools
import os
from gensim.corpora import dictionary, corpora
from gensim.models.tfidfmodel import TfidfModel


class CorpusModel(object):

    def __init__(self, cur, dictFile=None, corpusLoc=None):
        self.cur = cur
        self.corpus = self.loadCorpus()
        self.corpusLoc = corpusLoc
        if not dictFile:
            self.dictFile = "{}/tmp/corpusdict".format(os.getcwd())
            self.dictionary = self.buildDict()
            self.dictionary.filter_extremes(no_below=2, no_above=0.5,
                                            keep_n=100000)
            self.dictionary.compactify()

        else:
            self.dictFile = dictFile
            self.dictionary = self.loadDict()

    def getText(self):
        for item in self.cur:
            yield item['text'].split(), item['_id']

    def __iter__(self):
        for key, value in self.dictionary.iteritems():
            yield (key, value)

    def buildDict(self):
        self.docs, self.ids = itertools.izip(*self.getText())
        return dictionary.Dictionary(self.docs)

    def buildDoc2Bow(self):
        for text, id in itertools.izip(self.docs, self.ids):
            yield {id: self.dictionary.doc2bow(text)}

    def getTokenFreq(self):
        return self.dictionary.token2id

    def saveDict(self):
        self.dictionary.save(self.dictFile)

    def loadDict(self):
        return Dictionary.load(self.dictFile)

    def MakeTfidfModel(self):
        tfidf = TfidfModel()
        for doc in self.buildDoc2Bow():
            for docid, bow in doc.iteritems():
                yield tfidf[bow]

    def writeCorpus(self):
        tmpLoc = "{}/tmp/.format(os.getcwd()"
        error = "No {} folder/permission to write corpus.mm".format(tmpLoc)
        try:
            self.corpusLoc = '{}/tmp/corpus.mm'.format(os.getcwd())
        except:

            raise EnvironmentError(error)
        self.corpus = self.buildDoc2Bow()
        try:
            corpora.MmCorpus.serialize(self.corpusLoc, self.corpus)
        except:
            raise EnvironmentError(error)

    def loadCorpus(self):
        if self.corpusLoc:
            try:
                self.corpus = corpora.MmCorpus(self.corpusLoc)
            except:
                try:
                    self.writeCorpus()
                except:
                    pass
        else:
            try:
                self.writeCorpus()
            except:
                pass

if __name__ == '__main__':
    dataFile = "{}/tmp/bowdata.json".format(os.getcwd())
    dictFile = "{}/tmp/corpusdict".format(os.getcwd())
    cur = simMongoDb(n=10, array=True, jsonLoc=dataFile)
    model = CorpusModel(cur, dictFile=None)
    model.saveDict()
    model.MakeTfidfModel()
