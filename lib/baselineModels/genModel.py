from __future__ import print_function, absolute_import
from var.mongoSim import simMongoDb
import numpy as np
import itertools
import os
from gensim.corpora.dictionary import Dictionary


class CorpusModel(object):

    def __init__(self, cur, dictFile=None):
        self.cur = cur
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
        docs, _ = itertools.izip(*self.getText())
        return Dictionary(docs)

    def buildDoc2Bow(self):
        docs, ids = itertools.izip(*self.getText())
        for text, id in itertools.izip(docs, ids):
            yield {id: self.dictionary.doc2bow(text)}

    def getTokenFreq(self):
        return self.dictionary.token2id

    def saveDict(self):
        self.dictionary.save(self.dictfile)

    def loadDict(self):
        return Dictionary.load(self.dictFile)


if __name__ == '__main__':
    dataFile = "{}/tmp/bowdata.json".format(os.getcwd())
    dictFile = "{}/tmp/corpusdict".format(os.getcwd())
    cur = simMongoDb(n=10, array=True, jsonLoc=dataFile)
    model = CorpusModel(cur, dictFile=None)
    for item in model:
        print(item)
