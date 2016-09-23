#-------------------------------------------------------------------------------
# Name:         Text Stats
# Purpose:      Collect statistics on my text corpus
# Author:       Devon Muraoka
# Created:      9/20/16
# Copyright:   (c) Devon Muraoka, Bharat Ramanathan 
#-------------------------------------------------------------------------------
from __future__ import absolute_import, print_function
import os
import pprint
from collections import defaultdict
import settings
import glob
import logging
import string
import cPickle


cwd = os.path.join(settings.project_root, 'tmp')

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO, filename=cwd+'/trigram.log')
genres = ['Adventure', 'Angst', 'Drama', 'Family',
          'Fantasy', 'Friendship', 'Humor', 'Hurt-Comfort',
          'Romance', 'Sci-Fi', 'Supernatural']

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

STOPWORDS = frozenset(w for w in STOPWORDS.split() if w)

counts = {'total': {}}

def get_texts():
    for label in genres:
        logger.info('Working on {} Stories'.format(label))
        texts = glob.glob('{}/sample_dir/{}/*.txt'.format(cwd, label))
        legal_characters = list(string.ascii_lowercase)
        legal_characters.append(' ')
        legal_characters = set(legal_characters)
        for doc in texts:
            _id = doc.split('\\')[-1][:-4]
            with open(doc) as f:
                text = f.read()
            remove = list(set(text) - legal_characters)
            for letter in remove:
                text = text.replace(letter, ' ')
                text = ' '.join([word for word in text.split() if word not in STOPWORDS])
            logger.debug('Working on text:{}'.format(text))
            yield label, text, _id


def get_trigrams(text):
    for label, text, _id in get_texts():
        tokens = []
        for i in xrange(len(text) - 3):
            tokens.append(text[i:i + 3])
        yield label, tokens


def generate_ngrams(text, n):
    for i in xrange(len(text) - n):
        yield text[i:i + n]


def generate_ngram_frequencies(text, n):
    n_grams = {}
    n_gram_frequencies = {}
    for ngram in generate_ngrams(text, n):
        if ngram in n_grams:
            n_grams[ngram] += 1
        else:
            n_grams[ngram] = 1
    for key in n_grams:
        n_gram_frequencies[key] = float(n_grams[key]) / len(text)
    return n_gram_frequencies


def count_words():
    logger.info('Starting Word Counting')
    for label, grams in get_trigrams():
        for trigram in grams:
            if label in counts:
                if trigram in counts[label]:
                    counts[label][trigram] += 1
                else:
                    counts[label][trigram] = 1
            else:
                counts[label] = defaultdict()
                counts[label][trigram] = 1
            if trigram in counts['total']:
                counts['total'][trigram] += 1
            else:
                counts['total'][trigram] = 1

    logger.info('Writing Counts to File')
    cPickle.dump(counts, open(os.path.join(cwd, 'counts.dict'), 'wb'))
    pprint.pprint(counts)
    logger.info('Finished')

def calculate_frequency():
    freqs = {}#defaultdict(lambda: 0)
    for label, text, _id in get_texts():
        freq = generate_ngram_frequencies(text, 3)
        freqs[_id] = {}
        freqs[_id]['ngram_frequency'] = freq
        print(len(freqs))
        if len(freqs)%10000 == 0:
            cPickle.dump(freqs, open(os.path.join(cwd, 'trigram.freq'), 'wb'))
            freqs = defaultdict(lambda: 0)
            break





if __name__ == '__main__':
    #count_words()
    calculate_frequency()