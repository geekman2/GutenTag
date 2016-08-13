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
    text = getText(doc)
    if not getLang(text):
        print(doc['_id'])


if __name__ == '__main__':
    cur = getCursor()
    for i in range(0, 1000):
        item = cur.__getitem__(i)
        filterLang(item)
        # parsed = tokenize(getText(item))
        # for j in parsed[:10]:
        #    print j
