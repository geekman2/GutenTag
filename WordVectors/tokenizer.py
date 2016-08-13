# -------------------------------------------------------------------------------
# Name:         tokenizer.py
# Purpose:      Read data from MongoDb and tokenize using spacy
# Author:       Bharat Ramanathan
# Created:      08/12/2016
# Copyright:    (c) Bharat Ramanathan
# -------------------------------------------------------------------------------
from pymongo import MongoClient
from spacy.en import English


def getCursor(ip="159.203.187.28", port="27017"):
    # get a cursor object to iterate over
    db = MongoClient('mongodb://{}:{}'.format(ip, port))
    docs = db.data.fiction
    return docs.find({}, {'text': 1, '_id': 0})


def getText(cur):
    # return the text from the resulting document
    return cur["text"]


def tokenize(doc, parser=English()):
    # Intialize the parser and tokenize the text retrieved
    return parser(doc)

cur = getCursor()
for i in xrange(1, 100):
    item = cur.__getitem__(i)
    parsed = tokenize(getText(item))
    
# print the tokens in the sample
# tokenedSample = [tok for tok in parsedSample]
