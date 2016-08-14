# coding = utf-8
# ------------------------------------------------------------------------------
# Name:         tokenizer.py
# Purpose:      Parse and tokenize english text using spaCy
# Author:       Bharat Ramanathan
# Created:      08/14/2016
# Copyright:    (c) Bharat Ramanathan
# ------------------------------------------------------------------------------
from __future__ import print_function
import nltk
import multiprocessing
import parser


def tokenize(text, parser=nltk.wordpunct_tokenize):
    # Intialize the parser and tokenize the text retrieved
    return parser(text.lower())


def worker(doc):
    # Get the parsed data and update the document with tokenedText
    parsed = tokenize(doc['text'])
    parser.docs.update({'_id': doc['_id']},
                       {'$set': {'tokenedText': parsed}})


def tokenizeMultiple():
    # Multiprocessing-fu with the tokenization
    cur = parser.docs.find({'text': {'$exists': 'true'}}, {'text': 1})
    pool = multiprocessing.Pool(8)
    pool.map_async(worker, cur)
    pool.close()
    pool.join()

if __name__ == '__main__':
    tokenizeMultiple()
