# coding = utf-8
# ------------------------------------------------------------------------------
# Name:         parser.py
# Purpose:      Read data from MongoDb and tokenize using spacy
# Author:       Bharat Ramanathan
# Created:      08/12/2016
# Copyright:    (c) Bharat Ramanathan
# ------------------------------------------------------------------------------
from __future__ import print_function, absolute_import
import pymongo
import lib.WordVectors.language_filter
import multiprocessing
import var.settings as settings

if settings.debug == True:
    import time

docs = settings.docs

def notEnglish(doc):
    # Ids the documents that are non-English
    text = doc['text']
    if not lib.WordVectors.language_filter.is_english_nltk(text):
        return doc['_id']


def worker(item):
    # Filter through the documents and
    # remove those documents that have non-English text.
    _id = notEnglish(item)
    if _id:
        docs.remove({'_id': {'$in': [_id]}})


def removeNonEnglish():
    # Multiprocessing-fu with the non-English filter.
    if settings.debug == True:
        start = time.time()
    cur = docs.find({'text': {'$exists': 'true'}}, {'text': 1})
    pool = multiprocessing.Pool(8)
    pool.map_async(worker, cur)
    pool.close()
    pool.join()
    if settings.debug == True:
        print(docs.count())
        print("TOTAL RUNTIME:{}".format(time.time()-start))

if __name__ == '__main__':
    if settings.debug == True:
        print("START COUNT:{}".format(docs.count()))
    removeNonEnglish()
