# coding: utf8
from __future__ import print_function, absolute_import
from var.mongoSim import simMongoDb
# from pprint import pprint
import os
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec


class LabeledLineSentence(object):
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        for doc in self.source:
            yield LabeledSentence(**doc)


def prepData(data):
    for item in data:
        doc = item['tokenedText']
        flatdoc = [word for sent in doc for word in sent]
        cleandoc = [word.lower() for word in flatdoc if word.isalpha()]
        yield {'words': flatdoc, 'tags': [item['_id']]}


def TrainModel(sents):
    model = Doc2Vec(size=10000, window=10, min_count=5, workers=16, sample=1e5,
                    alpha=0.025, min_alpha=0.0001, negative=10)
    model.build_vocab(sents)
    for epoch in range(10):
        model.train(sents)
        model.alpha -= 0.002  # decrease the learning rate
    return model


if __name__ == '__main__':
    cur = simMongoDb(1)
    data = prepData(cur)
    sents = LabeledLineSentence(data)
    model = TrainModel(sents)
    modelPath = "{}/var/".format(os.getcwd())
    modelFile = modelPath+"trial.model"
    if os.path.exists(modelPath):
        model.save(modelFile)
    else:
        os.mkdir(modelPath)
        model.save(modelFile)
