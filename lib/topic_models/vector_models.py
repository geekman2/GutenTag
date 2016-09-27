# -*- coding: utf8 -*-
import os
import gensim

import settings
from lib.topic_models import pre_process

from time import time
import logging

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

STOPWORDS = """
a about above across after afterwards again against all almost alone along already also although always am among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere are around as at back be
became because become becomes becoming been before beforehand behind being below beside besides between beyond bill both bottom but by call can
cannot cant co computer con could couldnt cry de describe
detail did didn do does doesn doing don done down due during
each eg eight either eleven else elsewhere empty enough etc even ever every everyone everything everywhere except few fifteen
fify fill find fire first five for former formerly forty found four from front full further get give go
had has hasnt have he hence her here hereafter hereby herein hereupon hers herself him himself his how however hundred i ie
if in inc indeed interest into is it its itself keep last latter latterly least less ltd
just
kg km
made make many may me meanwhile might mill mine more moreover most mostly move much must my myself name namely
neither never nevertheless next nine no nobody none noone nor not nothing now nowhere of off
often on once one only onto or other others otherwise our ours ourselves out over own part per
perhaps please put rather re
quite
rather really regarding
same say see seem seemed seeming seems serious several she should show side since sincere six sixty so some somehow someone something sometime sometimes somewhere still such system take ten
than that the their them themselves then thence there thereafter thereby therefore therein thereupon these they thick thin third this those though three through throughout thru thus to together too top toward towards twelve twenty two un under
until up unless upon us used using
various very very via
was we well were what whatever when whence whenever where whereafter whereas whereby wherein whereupon wherever whether which while whither who whoever whole whom whose why will with within without would yet you
your yours yourself yourselves
"""
STOPWORDS = frozenset(w for w in STOPWORDS.split() if w)

class VectorModels(object):
    def __init__(self, data_loc, tmp_folder,
                 stopwords=STOPWORDS, sample_size=None):

        self.data_loc = data_loc
        self.tmp_folder = tmp_folder
        self.stopwords = stopwords

    def build_corpus(self, no_below=5, no_above=0.75):
        mycorpus = pre_process.MyCorpus(self.data_loc)
        mycorpus = self.filter_stopwords(mycorpus)

        mycorpus.dictionary.filter_extremes(no_below=no_below,
                                            no_above=no_above
                                            )

        mycorpus.dictionary.compactify()
        return mycorpus, mycorpus.dictionary

    def filter_stopwords(self, mycorpus):
        stopset = STOPWORDS

        stop_ids = [mycorpus.dictionary.token2id[stopword]
                    for stopword in stopset
                    if stopword in mycorpus.dictionary.token2id]

        remove_ids = [mycorpus.dictionary.token2id[token]
                      for token in mycorpus.dictionary.itervalues()
                      if len(token) < 3]

        mycorpus.dictionary.filter_tokens(stop_ids + remove_ids)
        mycorpus.dictionary.compactify()
        return mycorpus

    def load_corpus(self, no_below=15, no_above=0.5):
        corpus_file = os.path.join(self.tmp_folder, 'corpus.mm.index')
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
