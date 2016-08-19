from __future__ import print_function, absolute_import
from var.mongoSim import simMongoDb
import numpy as np
from itertools import izip
from os import getcwd
from gensim.corpora.dictionary import Dictionary


class CorpusModel(object):

    def __init__(self, cur):
        self.cur = cur
        self.dictionary = self.buildDict()
        self.dicionary.compactify()

    def getText(self):
        for item in self.cur:
            yield item['text'], item['_id']

    def __iter__(self):
        pass

    def buildDict(self):
        docs, _ = izip(*self.getText())
        return Dictionary(docs)

    def buildDoc2Bow(self):
        docs, ids = izip(*self.getText())
        for text, id in izip(docs, ids):
            yield {id: self.dictionary.doc2bow(text)}

    def getTokenFreq(self):
        return self.dictionary.token2id

    def saveDict()


if __name__ == '__main__':
    dataFile = "{}/var/bowdata.json".format(getcwd())
    cur = simMongoDb(n=10000, array=True, jsonLoc=dataFile)
    text, ids = izip(*getText(cur))
    model, features = makeModel(text, tfidf=True)
    cdists = getModelInfo(model, features)
    # print(cdists)
