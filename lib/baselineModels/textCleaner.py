from __future__ import print_function, absolute_import
import itertools
import nltk
# from var.mongoSim import simMongoDb
import lib.WordVectors.parser as mongoClient
import os
import json
import time
import gensim


class cleanText(object):

    def __init__(self, cur):
        self.cur = cur
        self.texts = None
        self.id = None
        self.stemmer = gensim.parsing.PorterStemmer()

    def tokenize(self):
        stops = set(nltk.corpus.stopwords.words("english"))
        for doc in self.texts:
            yield [token for token in
                   gensim.utils.tokenize(doc, lowercase=True,
                                         deacc=True, errors="ignore")
                   if token not in stops]

    def stem(self):
        for doc in self.tokenize():
            yield [self.stemmer.stem(token) for token in doc]

    def getText(self):
        for item in self.cur:
            yield item['text'], item['_id']

    def __iter__(self):
        self.texts, self.ids = itertools.izip(*self.getText())
        for item, idx in itertools.izip(self.stem(), self.ids):
            yield {'text': u' '.join(item),
                   '_id': str(idx)}


def writeCleanText(cur, outFile):
    with open(outFile, 'a') as f:
        for item in cleanText(cur):
            json.dump(item, f)
            f.write('\n')


if __name__ == '__main__':
    start = time.time()
    dataPath = "{}/tmp/testFiles".format(os.getcwd())
    docs = mongoClient.docs
    cur = docs.find({'text': {'$exists': 'true'}}, {'text': 1})
    # cur = simMongoDb(n=10000, array=True, dataLoc=dataPath)
    jsonPath = "{}/tmp/".format(os.getcwd())
    jsonFile = jsonPath+"genDataBig.json"
    # writeCleanText(cur[:500000], jsonFile)
    for item in cleanText(cur[:10]):
        print("i")
    print(time.time() - start)
