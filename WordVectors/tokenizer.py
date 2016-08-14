# coding = utf-8
# ------------------------------------------------------------------------------
# Name:         tokenizer.py
# Purpose:      Read data from MongoDb and tokenize using spacy
# Author:       Bharat Ramanathan
# Created:      08/12/2016
# Copyright:    (c) Bharat Ramanathan
# ------------------------------------------------------------------------------
from __future__ import print_function
from pymongo import MongoClient
from lnFilter import is_english_nltk,is_english_langid
import time
import multiprocessing

db_ip='159.203.187.28'
db_port='27017'
db = MongoClient('mongodb://{}:{}'.format(db_ip,db_port))
docs = db.data.fiction

def not_english(doc):
    text = doc['text']
    if not is_english_nltk(text):
        return doc['_id']

def worker(item):
    _id = not_english(item)
    if _id:
        docs.remove({'_id':{'$in':[_id]}})

def remove_non_english():
    start = time.time()
    cur = docs.find({'text': {'$exists': 'true'}}, {'text': 1})
    nonEng = []
    pool = multiprocessing.Pool(8)
    pool.map_async(worker,cur)
    pool.close()
    pool.join()
    print(docs.count())
    print("TOTAL RUNTIME:{}".format(time.time()-start))

if __name__ == '__main__':
    print("START COUNT:{}".format(docs.count()))
    remove_non_english()
