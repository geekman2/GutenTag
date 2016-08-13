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
        return doc['_id']


if __name__ == '__main__':
    cur = getCursor()
    nonEng = []
    for item in cur:
        nonEng.append(filterLang(item))
    nonEng = [item for item in nonEng if item is not None]
