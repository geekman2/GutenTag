from __future__ import print_function, absolute_import
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from var.mongoSim import simMongoDb
import numpy as np
from itertools import izip
from os import getcwd
from sklearn.metrics.pairwise import cosine_distances


def getText(cur):
    for item in cur:
        yield item['text'], item['_id']


def makeSmallModel(data, bow=False, tfidf=False, **kwargs):
    if not kwargs:
        if bow:
            kwargs = dict(analyzer="word", tokenizer=None,
                          preprocessor=None, stop_words=None,
                          min_df=2, max_features=20000,
                          ngram_range=(1, 3))
            model = CountVectorizer(**kwargs)
        elif tfidf:
            kwargs = dict(max_df=0.8, max_features=20000,
                          min_df=0.2, stop_words=None,
                          use_idf=True, tokenizer=None,
                          ngram_range=(1, 3))
            model = TfidfVectorizer(**kwargs)
        else:
            raise ValueError('Please specify a model to use')
    else:
        if bow:
            model = CountVectorizer(**kwargs)
        elif tfidf:
            model = TfidfVectorizer(**kwargs)
        else:
            raise ValueError('Please specify a model to use')
    feature_matrix = model.fit_transform(data)
    feature_matrix = feature_matrix.toarray()
    return model, feature_matrix


def getModelInfo(model, features):
    print("Shape of the transformed features = {}".format(features.shape))
    # Uncomment to info:
    # vocab = model.get_feature_names()
    # dist = np.sum(features, axis=0)
    # for tag, count in izip(vocab, dist):
    #     print("word = {}, frequency = {}".format(tag, count))
    return cosine_distances(features)



if __name__ == '__main__':
    dataFile = "{}/tmp/bowdata.json".format(getcwd())
    cur = simMongoDb(n=10, array=True, jsonLoc=dataFile)
    text, ids = izip(*getText(cur))
    model, features = makeSmallModel(text, tfidf=True)
    print(features)
    cdists = getModelInfo(model, features)
    # print(cdists)
