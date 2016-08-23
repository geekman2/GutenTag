from __future__ import print_function, absolute_import
from var.mongoSim import simMongoDb
import itertools
import os
import gensim
import time
import lib.baselineModels.textCleaner as cleaner
import lib.WordVectors.parser as mongoClient


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
        self.tfidfSimIndexFile = "{}/tmp/tfidf.index".format(cwd)
        self.ldaFile = "{}/tmp/ldaModel.mm".format(cwd)
        self.ldaSimIndexFile = "{}/tmp/lda.index".format(cwd)


class corpusModel(textModel):
    def __init__(self, cur):
        super(self.__class__, self).__init__(cur)
        self.loadDict()
        self.corpus = self.loadCorpus()
        self.tfidfCorpus = self.loadTfidfCorpus()
        self.ldaCorpus = self.loadLDAModel()

    def buildDoc2Bow(self):
        for doc in self.docs:
            yield self.dictionary.doc2bow(doc)

    def loadDict(self):
        try:
            self.dictionary =\
                gensim.corpora.dictionary.Dictionary.load(self.dictFile)
        except IOError:
            self.dictionary = gensim.corpora.dictionary.Dictionary(self.docs)
            self.dictionary.filter_extremes(no_below=5, no_above=0.5,
                                            keep_n=100000)
            self.dictionary.compactify()

            self.dictionary.save(self.dictFile)

    def loadCorpus(self):
        try:
            return gensim.corpora.mmcorpus.MmCorpus(self.corpusFile)
        except IOError:
            corpus = self.buildDoc2Bow()
            gensim.corpora.mmcorpus.MmCorpus.serialize(self.corpusFile, corpus)
            return corpus

    def loadTfidfCorpus(self):
        try:
            self.tfidfModel =\
                gensim.models.tfidfmodel.TfidfModel.load(self.tfidfFile)
            tfidfCorpus = self.tfidfModel[self.corpus]
            return tfidfCorpus
        except IOError:
            params = dict(corpus=self.corpus,
                          id2word=self.dictionary
                          )
            self.tfidfModel = gensim.models.tfidfmodel.TfidfModel(**params)
            tfidfCorpus = self.tfidfModel[self.corpus]
            self.tfidfModel.save(self.tfidfFile)
            return tfidfCorpus

    def loadLDAModel(self, tfidf=True):
        try:
            self.ldaModel =\
                gensim.models.ldamulticore.LdaMulticore.load(self.ldaFile)
            if tfidf:
                corpus = self.tfidfCorpus
            else:
                corpus = self.corpus
        except IOError:
            if tfidf:
                print(self.tfidfCorpus)
                params = dict(corpus=self.tfidfCorpus,
                              id2word=self.dictionary,
                              workers=16)
            else:
                params = dict(corpus=self.corpus,
                              id2word=self.dictionary,
                              workers=16)
            self.ldaModel = gensim.models.ldamulticore.LdaMulticore(**params)
            self.ldaModel.save(self.ldaFile)
        return self.ldaModel[corpus]


class SimilarityModel(corpusModel):
    def __init__(self, cur):
        super(self.__class__, self).__init__(self, cur)

    def loadSimIndex(self, type='lda', n=10):
        if type == 'lda':
            params = dict(fname=self.ldaSimIndexFile,
                          corpus=self.ldaCorpus,
                          num_best=n)
            try:
                self.similar_index =\
                    gensim.similarities.Similarity.load(self.ldaSimIndexFile)
            except IOError:
                self.similar_index =\
                    gensim.similarities.Similarity(**params)

    def getDocVec(self, doc, type='tfidf'):
        bow = self.dictionary.doc2bow(doc)
        if type == 'tfidf':
            return self.ldaModel[self.tfidfModel[bow]]
        elif type == 'lda':
            return self.ldaModel[bow]

    def find_similar(self, doc, n=10):
        vec = self.dictionary.doc2vec(doc)
        sims = self.similar_index[vec]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        for elem in sims[:n]:
            idx, value = elem
            print(' '.join(self.docs[idx]), value)


if __name__ == '__main__':
    dataFile = "{}/tmp/genData.json".format(os.getcwd())
    cur = simMongoDb(n=10, array=True, jsonLoc=dataFile)
    start = time.time()
    docs = mongoClient.docs
    # cur = docs.find({'text': {'$exists': 'true'}}, {'text': 1})
    # cursy = itertools.islice(cleaner.cleanText(cur[:5]), 5)
    theModel = corpusModel(cur)
    print(time.time() - start)
