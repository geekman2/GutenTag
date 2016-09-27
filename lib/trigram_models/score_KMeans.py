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
import settings
import cPickle
from bson import ObjectId
import itertools
import string
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd


logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

cwd = settings.project_root
working_directory = os.path.join(cwd, 'tmp')
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

random_seed = np.random.seed(1791)
with open(os.path.join(working_directory, 'KMeans.Cluster'),'rb') as f:
    model = cPickle.load(f)

genres = ['Family', 'Humor', 'Romance', 'Sci-Fi', 'Supernatural']
sample_docs = cPickle.load(open(os.path.join(working_directory, 'sample_docs.p'), 'rb'))
pprint(len(sample_docs))
docs = settings.docs.find({'_id': {'$in': sample_docs}},
                          projection={'trigram_frequency': 1})

legal_characters = list(string.ascii_lowercase)
legal_characters.append(' ')

all_keys = [''.join(x) for x in itertools.permutations(legal_characters, 3)]
stuffs = []
scores = []
for i, doc in enumerate(docs[80000:]):
    i += 1
    _id = doc["_id"]
    frequency = doc["trigram_frequency"]
    stuff = {trigram: round(frequency[trigram] * 10000, 3) if trigram in frequency else 0. for trigram in all_keys}
    stuff["_id"] = _id
    stuffs.append(stuff)
    if i % 10000 == 0:
        logger.info('Building Dataframe for batch #{} of {}'.format(i/10000, len(sample_docs)/10000))
        trigram_data = pd.DataFrame(stuffs)
        logger.info('Scoring')
        cluster_data = trigram_data.drop(["_id"], axis=1)
        score = silhouette_score(cluster_data, model.predict(cluster_data))
        print('score:', score)
        scores.append(score)
        stuffs = []

pprint(scores)
print(sum(scores)/10)