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

import settings
import glob
import logging
import string
import cPickle

cwd = os.path.join(settings.project_root, 'tmp')

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
genres = ['Adventure', 'Angst', 'Drama', 'Family',
          'Fantasy', 'Friendship', 'Humor', 'Hurt-Comfort',
          'Romance', 'Sci-Fi', 'Supernatural']

counts = {'total': {}}

def get_texts():
    for label in genres:
        logger.info('Working on {} Stories'.format(label))
        texts = glob.glob('{}/sample_dir/{}/*.txt'.format(cwd, label))
        legal_characters = list(string.ascii_lowercase)
        legal_characters.append(' ')
        legal_characters = set(legal_characters)
        for doc in texts:
            with open(doc) as f:
                text = f.read()
            remove = list(set(text) - legal_characters)
            for letter in remove:
                text = text.replace(letter, ' ')
                text = text.replace('  ', ' ')
            logger.debug('Working on text:{}'.format(text))
            yield label, text


def get_trigrams():
    for label, text in get_texts():
        tokens = []
        for i in xrange(len(text) - 3):
            tokens.append(text[i:i + 3])
        yield label, tokens

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
                counts[label] = {trigram: 1}
            if trigram in counts['total']:
                counts['total'][trigram] += 1
            else:
                counts['total'][trigram] = 1

    logger.info('Writing Counts to File')
    cPickle.dump(counts, open(os.path.join(cwd, 'counts.dict'), 'wb'))
    pprint.pprint(counts)
    logger.info('Finished')

if __name__ == '__main__':
    count_words()