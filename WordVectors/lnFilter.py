# coding: utf-8
# ------------------------------------------------------------------------------
# Name:         lnFilter.py
# Purpose:      Rundimentary language filter for text based on stopwords
# Author:       Bharat Ramanathan
# Created:      08/13/2016
# Copyright:    (c) Bharat Ramanathan
# ------------------------------------------------------------------------------
from __future__ import print_function
import nltk
import langid

# Get a set of english stop words.
# Example: [u'all', u'just', u'being', u'over', u'both', ...]
ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))

# Set of NON_ENGLISH_STOPWORDS
# Example u'ebben', u'negl', u'dere', u'noista', u'dazu', u'otra', ...]
NON_ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words()) - ENGLISH_STOPWORDS

STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang))
                  for lang in nltk.corpus.stopwords.fileids()
                  }


def is_english_nltk(text):
    # Identify the lanugage using the NLTK implementation.
    text = text.lower()  # convert to lowercase
    words = set(nltk.wordpunct_tokenize(text))
    return len(words & ENGLISH_STOPWORDS) > len(words & NON_ENGLISH_STOPWORDS)


def is_english_langid(text):
    # Identify language using langid
    text = text.lower()
    lang = langid.classify(text)
    return lang[0] == 'en'

if __name__ == '__main__':
    test1 = u"This is an english"
    test2 = u"Ceci est français"
    test3 = u"Questo è italiano"
    test4 = u"dit is nederlands"
    test5 = u"to jest polski"

    for item in [test1, test2, test3, test4, test5]:
        # print isEnglish(item)
        print(is_english_langid(item))
