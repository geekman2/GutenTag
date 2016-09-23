# coding = utf-8
# ------------------------------------------------------------------------------
# Name:         Create corpus models
# Purpose:      Create corpus models#TODO improve this description
# Author:       Bharat Ramanathan, Devon Muraoka
# Created:      9/6/2016
# Copyright:    (c) Bharat Ramanathan, Devon Muraoka
# ------------------------------------------------------------------------------
from __future__ import print_function, absolute_import
import settings
import os
import logging
import gensim
import string
import glob


logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

cwd = settings.project_root
working_directory = os.path.join(cwd, 'tmp', 'modeldir')
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

#genres = ['Adventure', 'Angst', 'Drama', 'Family',
#          'Fantasy', 'Friendship', 'Humor', 'Hurt-Comfort',
#          'Romance', 'Sci-Fi', 'Supernatural']

genres = ['Family', 'Humor', 'Romance', 'Sci-Fi', 'Supernatural']

STOPWORDS = """
a about above across after afterwards again against all almost alone along already also although always am among amongst
amoungst amount an and another any anyhow anyone anything anyway anywhere are around as at back be
became because become becomes becoming been before beforehand behind being below beside besides between beyond bill both
 bottom but by call can
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
same say see seem seemed seeming seems serious several she should show side since sincere six sixty so some somehow
someone something sometime sometimes somewhere still such system take ten
than that the their them themselves then thence there thereafter thereby therefore therein thereupon these they thick
thin third this those though three through throughout thru thus to together too top toward towards twelve twenty two un
under
until up unless upon us used using
various very very via
was we well were what whatever when whence whenever where whereafter whereas whereby wherein whereupon wherever
whether which while whither who whoever whole whom whose why will with within without would yet you
your yours yourself yourselves
"""

class CorpusModel(object):
    def get_texts(self):
        for label in genres:
            logger.info('Working on {} Stories'.format(label))
            texts = glob.glob('{}/tmp/sample_dir/{}/*.txt'.format(cwd, label))
            logger.debug(texts)
            legal_characters = list(string.ascii_lowercase)
            legal_characters.append(' ')
            legal_characters = set(legal_characters)
            for doc in texts:
                with open(doc) as f:
                    text = f.read()
                remove = list(set(text) - legal_characters)
                for letter in remove:
                    text = text.replace(letter, ' ')
                    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
                logger.debug('Working on text:{}'.format(text))
                yield text

    def get_trigrams(self):
        for text in self.get_texts():
            tokens = []
            for i in xrange(len(text) - 3):
                tokens.append(text[i:i+3])
            yield tokens

    def load_dict(self):
        """
        :return:
        """
        self.dict_file = os.path.join(working_directory, 'corpus.dict')
        if os.path.isfile(self.dict_file):
            logger.info('Loading corpus dictionary from file')
            self.dictionary = gensim.corpora.dictionary.Dictionary.load(self.dict_file)
            logger.info('Corpus dictionary loaded from file')
        else:
            self.dictionary = gensim.corpora.Dictionary(self.get_trigrams())
            self.dictionary.filter_extremes(no_below=1, no_above=0.99, keep_n=100000)
            self.dictionary.compactify()
            self.dictionary.save(self.dict_file)
        return self.dictionary

    def get_bow(self):
        for doc in self.get_trigrams():
            yield self.dictionary.doc2bow(doc)

    def load_corpus(self):
        """
        Load corpus if it exists, else build corpus from docs
        Corpus being defined as the vector bag of words representation of all documents
        :rtype: object
        :return: Corpus
        """
        self.corpus_file = os.path.join(working_directory, 'corpora.mm')
        if os.path.isfile(self.corpus_file):
            logger.info('Trying to load corpus from disk')
            self.corpus = gensim.corpora.mmcorpus.MmCorpus(self.corpus_file)
            logger.info('loaded corpus from disk')
        else:
            logger.info('Corpus not found on disk. Building corpus and serializing to disk')
            gensim.corpora.mmcorpus.MmCorpus.serialize(self.corpus_file, self.get_bow(), id2word=self.dictionary)
            logger.info('Successfully built corpus and saved corpus to disk')
            self.corpus = gensim.corpora.mmcorpus.MmCorpus(self.corpus_file)
        return self.corpus

    def load_tfidf_corpus(self):
        """
        Load TFIDF model from disk, if it does not exist, train and save it to disk
        :return: TFIDF Corpus
        """
        self.tfidf_file = os.path.join(working_directory, 'tfidfCorpora.mm')
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


if __name__ == '__main__':
    mycorpus = CorpusModel()
    mycorpus.load_dict()
    mycorpus.load_corpus()
    mycorpus.load_tfidf_corpus()