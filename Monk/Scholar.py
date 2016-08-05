# -------------------------------------------------------------------------------
# Name:         Monk
# Purpose:      Parse data to and from MongoDB
# Author:      Devon Muraoka
# Created:     7/22/2016
# Copyright:   (c) Devon Muraoka 
# -------------------------------------------------------------------------------
import pymongo
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import itertools
from scipy.spatial.distance import cityblock

db_client = pymongo.MongoClient('mongodb://localhost:27017')

random_seed = np.random.seed(1791)
examples = db_client.test
docs = list(examples.grams.find({"trigram_frequency": {"$exists": True}}))
df = pd.DataFrame()
all_keys = set()

for d in docs:
    all_keys.update(d['trigram_frequency'].keys())

print "NUMBER OF UNIQUE TRIGRAMS", len(all_keys)

stuffs = []
for i, doc in enumerate(docs):
    doc = doc["trigram_frequency"]
    stuff = {trigram: round(doc[trigram] * 10000, 3) if trigram in doc else 0. for trigram in all_keys}
    stuffs.append(stuff)

count = 0
for j in stuffs:
    for k in j:
        if j[k] != 0:
            count += 1

print "NON-ZERO FREQUENCY TRIGRAM COUNT:", count

data = pd.DataFrame(stuffs)
log_data = np.log(data)

clusterer = KMeans(n_clusters=100, random_state=random_seed).fit(data)
search = data.loc[10].reshape(1, -1)
print clusterer.predict(search)
db_client.close()
