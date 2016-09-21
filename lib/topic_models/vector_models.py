# -*- coding: utf8 -*-
import os
import gensim
from nltk.corpus import stopwords


import settings
from lib.topic_models import pre_process

from time import time
import logging

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

additional_stops = pre_process.STOPWORDS

class VectorModels(object):
    def __init__(self, data_loc, tmp_folder,
                 stopwords=additional_stops, sample_size=None):

        self.data_loc = data_loc
        self.tmp_folder = tmp_folder
        self.stopwords = stopwords

    def build_corpus(self, no_below=5, no_above=0.75):
        mycorpus = pre_process.MyCorpus(self.data_loc)
        mycorpus = self.filter_stopwords(mycorpus, self.stopwords)

        mycorpus.dictionary.filter_extremes(no_below=no_below,
                                            no_above=no_above
                                            )

        mycorpus.dictionary.compactify()
        return mycorpus, mycorpus.dictionary

    def filter_stopwords(self, mycorpus,
                         additional_stopwords=additional_stops):
        stopset = set(stopwords.words('english')).union(additional_stopwords)

        stop_ids = [mycorpus.dictionary.token2id[stopword]
                    for stopword in stopset
                    if stopword in mycorpus.dictionary.token2id]

        remove_ids = [mycorpus.dictionary.token2id[token]
                      for token in mycorpus.dictionary.itervalues()
                      if len(token) < 3]

        mycorpus.dictionary.filter_tokens(stop_ids + remove_ids)
        mycorpus.dictionary.compactify()
        return mycorpus

    def load_corpus(self, no_below=5, no_above=0.75):
        corpus_file = os.path.join(self.tmp_folder, 'corpus.mm')
        dictionary_file = os.path.join(self.tmp_folder, 'corpus.dict')
        if os.path.isfile(corpus_file) and os.path.isfile(dictionary_file):
            mycorpus = gensim.corpora.mmcorpus.MmCorpus(corpus_file)
            dictionary = gensim.corpora.Dictionary.load(dictionary_file)
        else:
            mycorpus, dictionary = self.build_corpus(no_below, no_above)
            dictionary.save(dictionary_file)

            gensim.corpora.MmCorpus.serialize(corpus_file,
                                              mycorpus,
                                              id2word=dictionary
                                              )
        return mycorpus, dictionary

    def build_tfidf_model(self, dictionary):
        tfidf_file = os.path.join(self.tmp_folder, 'tfidf_model.mm')
        if os.path.isfile(tfidf_file):
            tfidf_model = gensim.models.TfidfModel.load(tfidf_file)
        else:
            tfidf_model = gensim.models.TfidfModel(id2word=dictionary,
                                                   dictionary=dictionary
                                                   )
            tfidf_model.save(tfidf_file)
        return tfidf_model

    def build_tfidf_corpus(self, corpus, dictionary):
        tfidf_file = os.path.join(self.tmp_folder, 'tfidf_corpus.mm')
        if os.path.isfile(tfidf_file):
            tfidf_corpus = gensim.corpora.mmcorpus.MmCorpus(tfidf_file)
        else:
            tfidf_model = self.build_tfidf_model(dictionary)
            tfidf_corpus = tfidf_model[corpus]
            gensim.corpora.MmCorpus.serialize(tfidf_file, tfidf_corpus)
        return tfidf_corpus


if __name__ == '__main__':
    # Build the Corpus
    start_corpus = time()
    data_loc = os.path.join(settings.project_root, 'tmp', 'text_corpus.dat')
    tmp_folder = os.path.join(settings.project_root, 'tmp', 'modeldir')

    vectors = VectorModels(data_loc, tmp_folder)
    corpus, dictionary = vectors.load_corpus()
    vectors.build_tfidf_corpus(corpus, dictionary)

    stop_corpus = time()
    corpus_time = round(stop_corpus - start_corpus, 3)
    logger.info('time taken to build corpus = {}s'.format(corpus_time))
