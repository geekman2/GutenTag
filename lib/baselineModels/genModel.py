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


class CorpusModel(object):
    def __init__(self, cursor, lda_topics=14):
        self.cur = cursor
        # Configure Files
        cwd = os.getcwd()
        working_directory = '{}/tmp/modeldir/'.format(cwd)
        if not os.path.exists(working_directory):
            os.makedirs(working_directory)
        self.dict_file = '{}corpus.dict'.format(working_directory)
        self.corpus_file = '{}corpora.mm'.format(working_directory)
        self.tfidf_file = '{}tfidfCorpora.mm'.format(working_directory)
        self.tfidf_sim_indexFile = '{}tfidf.index'.format(working_directory)
        self.lda_file = '{}ldaModel.mm'.format(working_directory)
        self.sim_index_file = '{}simFile.idx'.format(working_directory)
        self.simIndexPrefix = '{}simidx'.format(working_directory)
        # Train Model
        self.docs, self.idx = itertools.izip(*self.get_text())
        self.dictionary = self.load_dict()
        self.corpus = self.load_corpus()
        self.tfidf_corpus = self.load_tfidf_corpus()
        self.lda_corpus = self.load_lda_model(n_topics=lda_topics)

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
            corpus = gensim.corpora.mmcorpus.MmCorpus(self.corpus_file)
            logger.info('loaded corpus from disk')
        else:
            logger.info('Corpus not found on disk. Building corpus and serializing to disk')
            gensim.corpora.mmcorpus.MmCorpus.serialize(self.corpus_file, self.build_doc_2_bow())
            logger.info('Successfully built corpus and saved corpus to disk')
            self.corpus = gensim.corpora.mmcorpus.MmCorpus(self.corpus_file)
        return self.corpus

    def load_tfidf_corpus(self):
        """
        Load TFIDF model from disk, if it does not exist, train and save it to disk
        :return: TFIDF Corpus
        """
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

    def load_lda_model(self, n_topics=14):
        """
        Load Latent Dirichlet Allocation Model into memory, if none exists, create it
        :param n_topics: The number of topics the LDA algorithm will pick out
        :return: 
        """
        if os.path.isfile(self.lda_file):
            logger.info('trying to load LDA model from disk')
            self.lda_model = \
                gensim.models.ldamulticore.LdaMulticore.load(self.lda_file)
            logger.info('loaded LDA model from disk')
        else:
            logger.info('training LDA model')
            params = {'corpus': self.tfidf_corpus,
                      'id2word': self.dictionary,
                      'num_topics': n_topics,
                      'workers': 16
                      }
            self.lda_model = gensim.models.ldamulticore.LdaMulticore(**params)
            logger.info('Saving LDA model to disk')
            self.lda_model.save(self.lda_file)
            logger.info('Saved LDA model to disk')
        self.lda_corpus = self.lda_model[self.tfidf_corpus]
        return self.lda_corpus

    def load_sim_index(self, n_features=14, n_best=10):
        """
        Return index of similarity
        :param n_features:
        :param n_best: Number of documents to return, sorted by most similar
        :return:Similar Object:
        """
        params = {'output_prefix': self.simIndexPrefix,
                  'corpus': self.lda_corpus,
                  'num_features': n_features,
                  'num_best': 10
                  }
        if os.path.isfile(self.sim_index_file):
            logger.info('Loading Sim Index from disk')
            self.similar_index = gensim.similarities.docsim.Similarity.load(self.sim_index_file)
            logger.info('Loaded Sim Index from disk')
        else:
            logger.info('Creating Sim Index')
            self.similar_index = gensim.similarities.docsim.Similarity(**params)
            logger.info('Saving Sim Index to disk')
            self.similar_index.save(self.sim_index_file)
            logger.info('Saved Sim Index to disk')
        return self.similar_index


if __name__ == '__main__':
    # dataFile = '{}/tmp/genData.json'.format(os.getcwd())
    # cursy = simMongoDb(n=10, array=True, jsonLoc=dataFile)
    start = time.time()
    docs = mongoClient.docs
    cur = docs.find({'text': {'$exists': 'true'}}, {'text': 1})
    cursy = cleaner.cleanText(cur[:100])
    the_model = CorpusModel(cursy)
    the_model.load_sim_index()
    for sims in the_model.similar_index:
        print('sims = {}'.format(sims))
    # print(theModel.ldaModel.print_topics())
    print(time.time() - start)
