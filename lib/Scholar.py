# -------------------------------------------------------------------------------
# Name:         Monk
# Purpose:      Parse data to and from MongoDB
# Author:      Devon Muraoka
# Created:     7/22/2016
# Copyright:   (c) Devon Muraoka
# -------------------------------------------------------------------------------
from __future__ import print_function, absolute_import

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

import settings as settings

random_seed = np.random.seed(1791)
data = settings.db.data.fiction
clusterer = MiniBatchKMeans(n_clusters=14, random_state=random_seed)

logger = logging.getLogger('Scholar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

for i in xrange(50):
    logger.info("STARTING SAMPLE #{}".format(i))
    docs = list(data.aggregate([{"$sample": {"size": 1000}}], allowDiskUse=True))
    df = pd.DataFrame()
    all_keys = set()

    for d in docs:
        all_keys.update(d['trigram_frequency'].keys())

    logger.info("NUMBER OF UNIQUE TRIGRAMS:{}".format(len(all_keys)))

    stuffs = []
    for i, doc in enumerate(docs):
        _id = doc["_id"]
        _url = doc["url"]
        frequency = doc["trigram_frequency"]
        stuff = {trigram: round(frequency[trigram] * 10000, 3)
                 if trigram in frequency else 0. for trigram in all_keys}
        stuff["_id"] = _id
        stuff["_url"] = _url
        stuffs.append(stuff)

    trigram_data = pd.DataFrame(stuffs)

    cluster_data = trigram_data.drop(["_url", "_id"], axis=1)
    clusterer.partial_fit(cluster_data)

labels = clusterer.labels_
print(len(labels))
print(labels)
print(labels[10])


