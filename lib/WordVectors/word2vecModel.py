# coding: utf8
from __future__ import print_function, absolute_import
from var.mongoSim import simMongoDb
from itertools import chain
# from pprint import pprint
# from gensim.models import Word2Vec


def foo(word):
    if word.isalpha():
        return word.lower()
    else:
        return None


def prepData(data):
    for item in data:
        doc = item['tokenedText']
        for sent in doc:
            for i, word in enumerate(sent):
                sent[i] = foo(word)
                if sent[i] is None:
                    del sent[i]
        yield(doc)



if __name__ == '__main__':
    cur = simMongoDb(2)
    data = chain(prepData(cur))
    for item in data:
        yield item
