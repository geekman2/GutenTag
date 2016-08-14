# coding = utf-8
# ------------------------------------------------------------------------------
# Name:         parser.py
# Purpose:      Parse and tokenize english text using spaCy
# Author:       Bharat Ramanathan
# Created:      08/13/2016
# Copyright:    (c) Bharat Ramanathan
# ------------------------------------------------------------------------------
from __future__ import print_function
import nltk
import multiprocessing
import tokenizer


def tokenize(text, parser=nltk.wordpunct_tokenize):
    # Intialize the parser and tokenize the text retrieved
    return parser(text.lower())


def worker(doc):
    parsed = tokenize(doc['text'])
    tokenizer.docs.update({'_id': doc['_id']},
                          {'$set': {'tokenedText': parsed}})


def tokenizeMultiple():
    cur = tokenizer.docs.find({'text': {'$exists': 'true'}}, {'text': 1})
    pool = multiprocessing.Pool(8)
    pool.map_async(worker, cur)
    pool.close()
    pool.join()

if __name__ == '__main__':
    tokenizeMultiple()
