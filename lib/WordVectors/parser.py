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
import WordVectors.lnFilter
import multiprocessing
# import time #UNCOMMENT FOR DEBUGGING

# Necessary connection variables.
# db_ip = '159.203.187.28'
db_ip = 'localhost'
db_port = '27017'
db = pymongo.MongoClient('mongodb://{}:{}'.format(db_ip, db_port))
docs = db.data.fiction


def notEnglish(doc):
    # Ids the documents that are non-English
    text = doc['text']
    if not WordVectors.lnFilter.isEnglishNltk(text):
        return doc['_id']


def worker(item):
    # Filter through the documents and
    # remove those documents that have non-English text.
    _id = notEnglish(item)
    if _id:
        docs.remove({'_id': {'$in': [_id]}})


def removeNonEnglish():
    # Multiprocessing-fu with the non-English filter.
    # start = time.time() (uncomment for debugging-info)
    cur = docs.find({'text': {'$exists': 'true'}}, {'text': 1})
    pool = multiprocessing.Pool(8)
    pool.map_async(worker, cur)
    pool.close()
    pool.join()
    # uncomment the below lines for debugging info
    # print(docs.count())
    # print("TOTAL RUNTIME:{}".format(time.time()-start))

if __name__ == '__main__':
    # uncomment the below lines for debugging info
    # print("START COUNT:{}".format(docs.count()))
    removeNonEnglish()
