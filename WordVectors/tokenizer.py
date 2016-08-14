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
from lnFilter import getLang


def getCursor(ip="159.203.187.28", port="27017"):
    # get a cursor object to iterate over
    db = MongoClient('mongodb://{}:{}'.format(ip, port))
    docs = db.data.fiction
    return docs.find({'text': {'$exists': 'true'}}, {'text': 1})


def getText(doc):
    # return the text from the resulting document
    return doc['text']


def filterLang(doc):
    # find and return id for those documents that are non-english
    text = getText(doc)
    if not getLang(text):
        return doc['_id']


def getNonEnglish(cur):
    nonEng = []
    for item in cur[:10000]:
        nonEng.append(filterLang(item))
    nonEng = [item for item in nonEng if item is not None]
    return nonEng

if __name__ == '__main__':
    cur = getCursor()
    nonEngs = getNonEnglish(cur)
    print(nonEngs)
