# coding = utf-8
# ------------------------------------------------------------------------------
# Name:         tokenizer.py
# Purpose:      Parse and tokenize english text using spaCy
# Author:       Bharat Ramanathan
# Created:      08/14/2016
# Copyright:    (c) Bharat Ramanathan
# ------------------------------------------------------------------------------
from __future__ import print_function, absolute_import
import parser
from spacy.en import English
from time import time
from itertools import izip


def getText(cur):
    for item in cur:
        yield item['text'], item['_id']


def tokenize(texts, parser=English()):
    for doc in parser.pipe(texts, n_threads=16):
        yield [token.text for token in doc]


def writeText(cur):
    texts, ids = izip(*getText(cur))
    for text, ids in izip(tokenize(texts), ids):
        # print(text, ids) # - Uncomment for debug info.
        # parser.docs.update_one({'_id': doc['_id']},
        #                       {'$set': {'tokenedText': parsedList}})
        # DELETE THIS BRACE
        # '$unset':{'text':''}}) UNCOMMENTING WILL DELETE THE TEXT FIELD


if __name__ == '__main__':
    data = parser.docs
    cur = data.find({'text': {'$exists': 'true'}}, {'text': 1})
    start = time()
    writeText(cur[:10000])
    print(time()-start)
