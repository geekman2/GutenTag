#-------------------------------------------------------------------------------
# Name:         module1
# Purpose:
# Author:       Devon Muraoka
# Created:      9/22/16
# Copyright:   (c) Devon Muraoka, Bharat Ramanathan 
#-------------------------------------------------------------------------------
from __future__ import absolute_import, print_function
from pprint import pprint
import logging
import os

from sklearn.metrics import silhouette_score

import settings
import cPickle
from bson import ObjectId
import itertools
import string
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import random

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

cwd = settings.project_root
working_directory = os.path.join(cwd, 'tmp')
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

random_seed = np.random.seed(1791)
clusterer = MiniBatchKMeans(n_clusters=5, random_state=random_seed, batch_size=1000)

genres = ['Family', 'Humor', 'Romance', 'Sci-Fi', 'Supernatural']
sample_dict = cPickle.load(open(working_directory+'\\genre_mapping', 'rb'))
sample_docs = []
for x in genres:
    sample_docs.extend([ObjectId(y[:-4]) for y in sample_dict[x]])
scores = []
for _ in range(5):
    random.shuffle(sample_docs)
    cPickle.dump(sample_docs, open(os.path.join(working_directory, 'sample_docs.p'), 'wb'))
    docs = settings.docs.find({'_id': {'$in': sample_docs}},
                              projection={'trigram_frequency': 1})

    legal_characters = list(string.ascii_lowercase)
    legal_characters.append(' ')

    all_keys = [''.join(x) for x in itertools.permutations(legal_characters, 3)]
    stuffs = []

    for i, _id in enumerate(sample_docs[:80000]):
        i += 1
        doc = settings.docs.find({'_id': _id})[0]
        _id = doc["_id"]
        frequency = doc["trigram_frequency"]
        stuff = {trigram: round(frequency[trigram] * 10000, 3) if trigram in frequency else 0. for trigram in all_keys}
        stuff["_id"] = _id
        stuffs.append(stuff)
        if i % 10000 == 0:
            logger.info('Building Dataframe for batch #{} of {}'.format(i/10000, 80000/10000))
            trigram_data = pd.DataFrame(stuffs)
            logger.info('Beginning Clustering')
            cluster_data = trigram_data.drop(["_id"], axis=1)
            clusterer.partial_fit(cluster_data)
            stuffs = []
    cPickle.dump(clusterer, open('KMeans.Cluster', 'wb'))

    stuffs = []

    docs = settings.docs.find({'_id': {'$in': sample_docs}},
                              projection={'trigram_frequency': 1})
    for i, doc in enumerate(docs[80000:]):
        i += 1
        _id = doc["_id"]
        frequency = doc["trigram_frequency"]
        stuff = {trigram: round(frequency[trigram] * 10000, 3) if trigram in frequency else 0. for trigram in all_keys}
        stuff["_id"] = _id
        stuffs.append(stuff)
        if i % 10000 == 0:
            logger.info('Building Dataframe for batch #{} of {}'.format(i/10000, 20000/10000))
            trigram_data = pd.DataFrame(stuffs)
            logger.info('Scoring')
            cluster_data = trigram_data.drop(["_id"], axis=1)
            preds = clusterer.predict(cluster_data)
            #preds = [np.random.randint(0, 4)] * cluster_data.shape[0]
            #pprint(preds)
            score = silhouette_score(cluster_data, clusterer.predict(cluster_data))
            print('score:', score)
            scores.append(score)
            stuffs = []

pprint(scores)
print(sum(scores)/len(scores))

