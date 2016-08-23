from __future__ import print_function, absolute_import
# from var.mongoSim import simMongoDb
import itertools
# import numpy as np
import os
import gensim
import time
import lib.baselineModels.textCleaner as cleaner
import lib.WordVectors.parser as mongoClient
import logging

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class corpusModel(object):
    def __init__(self, cur):
        self.cur = cur

    def getText(self):
        for item in self.cur:
            yield item['text'].split(), item['_id']

    def _generateConf(self):
        cwd = os.getcwd()
        workingdir = "{}/tmp/modeldir/".format(cwd)
        if not os.path.exists(workingdir):
            os.makedirs(workingdir)
        self.dictFile = "{}corpus.dict".format(workingdir)
        self.corpusFile = "{}corpora.mm".format(workingdir)
        self.tfidfFile = "{}tfidfCorpora.mm".format(workingdir)
        self.tfidfSimIndexFile = "{}tfidf.index".format(workingdir)
        self.ldaFile = "{}ldaModel.mm".format(workingdir)
        self.simIndexFile = "{}simFile.idx".format(workingdir)
        self.simIndexPrefix = "{}simidx".format(workingdir)

    def trainModel(self, lda_topics=14):
        self.docs, self.idx = itertools.izip(*self.getText())
        self._generateConf()
        self.dictionary = self.loadDict()
        self.corpus = self.loadCorpus()
        self.tfidfCorpus = self.loadTfidfCorpus()
        self.ldaCorpus = self.loadLDAModel(n_topics=lda_topics)

    def buildDoc2Bow(self):
        return [self.dictionary.doc2bow(doc) for doc in self.docs]

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
        return self.dictionary

    def loadCorpus(self):
        try:
            logger.info("trying to load corpus from disk")
            corpus = gensim.corpora.mmcorpus.MmCorpus(self.corpusFile)
            logger.info("loaded corpus from disk")
        except IOError:
            logger.info("Building a corpus")

            logger.info("Saving corpus to disk")
            gensim.corpora.mmcorpus.MmCorpus.serialize(self.corpusFile,
                                                       self.buildDoc2Bow())
            logger.info("Saved corpus to disk")
            corpus = gensim.corpora.mmcorpus.MmCorpus(self.corpusFile)
        return corpus

    def loadTfidfCorpus(self):
        try:
            logger.info("trying to load tfidf model from disk")
            self.tfidfModel =\
                gensim.models.tfidfmodel.TfidfModel.load(self.tfidfFile)
            logger.info("loaded tfidf model from disk")
        except IOError:
            logger.info("training tfidf model")
            params = {"corpus": self.corpus,
                      "id2word": self.dictionary
                      }
            self.tfidfModel = gensim.models.tfidfmodel.TfidfModel(**params)
            logger.info("Saving tfidf model to disk")
            self.tfidfModel.save(self.tfidfFile)
            logger.info("Saved tfidf model to disk")
        self.tfidfCorpus = self.tfidfModel[self.corpus]
        return self.tfidfCorpus

    def loadLDAModel(self, n_topics=14):
        try:
            logger.info("trying to load LDA model from disk")
            self.ldaModel =\
                gensim.models.ldamulticore.LdaMulticore.load(self.ldaFile)
            logger.info("loaded LDA model from disk")
        except IOError:
            logger.info("training LDA model")
            params = {"corpus": self.tfidfCorpus,
                      "id2word": self.dictionary,
                      "num_topics": n_topics,
                      "workers": 16
                      }
            self.ldaModel = gensim.models.ldamulticore.LdaMulticore(**params)
            logger.info("Saving LDA model to disk")
            self.ldaModel.save(self.ldaFile)
            logger.info("Saved LDA model to disk")
        self.ldaCorpus = self.ldaModel[self.tfidfCorpus]
        return self.ldaCorpus

    def loadSimIndex(self, n_features=14, n_best=10):
        params = {"output_prefix": self.simIndexPrefix,
                  "corpus": self.ldaCorpus,
                  "num_features": n_features,
                  "num_best": 10
                  }
        try:
            logger.info("Loading Sim Index from disk")
            self.similar_index =\
                gensim.similarities.docsim.Similarity.load(self.simIndexFile)
            logger.info("Loaded Sim Index from disk")
        except IOError:
            logger.info("Creating Sim Index")
            self.similar_index =\
                gensim.similarities.docsim.Similarity(**params)
            logger.info("Saving Sim Index to disk")
            self.similar_index.save(self.simIndexFile)
            logger.info("Saved Sim Index to disk")
        return self.similar_index


if __name__ == '__main__':
    # dataFile = "{}/tmp/genData.json".format(os.getcwd())
    # cursy = simMongoDb(n=10, array=True, jsonLoc=dataFile)
    start = time.time()
    docs = mongoClient.docs
    cur = docs.find({'text': {'$exists': 'true'}}, {'text': 1})
    cursy = cleaner.cleanText(cur[:100])
    theModel = corpusModel(cursy)
    theModel.trainModel()
    theModel.loadSimIndex()
    for sims in theModel.similar_index:
        print("sims = {}".format(sims))
    # print(theModel.ldaModel.print_topics())
    print(time.time() - start)
