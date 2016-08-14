# coding = utf-8
# ------------------------------------------------------------------------------
# Name:         tokenizer.py
# Purpose:      Parse and tokenize english text using spacy
# Author:       Bharat Ramanathan, Devon Muraoka
# Created:      08/14/2016
# Copyright:    (c) Bharat Ramanathan, Devon Muraoka
# ------------------------------------------------------------------------------
from __future__ import print_function
import nltk
import multiprocessing
import parser


def tokenize(text, parser=nltk.wordpunct_tokenize):
    # Intialize the parser and tokenize the text retrieved
    return parser(text.lower())


def worker(doc):
    parsed = tokenize(doc['text'])
    parser.docs.update({'_id': doc['_id']},
                       {'$set': {'tokenedText': parsed}},
                       {'$unset':{'text':''}
                        })


def tokenizeMultiple():
    cur = parser.docs.find({'text': {'$exists': 'true'}}, {'text': 1})
    pool = multiprocessing.Pool(8)
    pool.map_async(worker, cur[:100])
    pool.close()
    pool.join()

if __name__ == '__main__':
    tokenizeMultiple()
