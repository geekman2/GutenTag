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
from sklearn.cluster import MiniBatchKMeans

db_client = pymongo.MongoClient('mongodb://localhost:27017')

random_seed = np.random.seed(1791)
data = db_client.data.fiction
docs = data.aggregate([{"$sample":{"size":5000}}],allowDiskUse=True)
df = pd.DataFrame()
all_keys = set()

for d in docs:
    all_keys.update(d['trigram_frequency'].keys())

print "NUMBER OF UNIQUE TRIGRAMS", len(all_keys)

stuffs = []
for i, doc in enumerate(docs):
    _id = doc["_id"]
    _url = doc["url"]
    frequency = doc["trigram_frequency"]
    stuff = {trigram: round(frequency[trigram] * 10000, 3) if trigram in frequency else 0. for trigram in all_keys}
    stuff["_id"] = _id
    stuff["_url"] = _url
    stuffs.append(stuff)

count = 0
for j in stuffs:
    for k in j:
        if j[k] != 0:
            count += 1

print "NON-ZERO FREQUENCY TRIGRAM COUNT:", count

data = pd.DataFrame(stuffs)

cluster_data = data.drop(["_url","_id"],axis=1)
try:
    print cluster_data["_url"]
except:
    print "Never mind, no url here"
clusterer = MiniBatchKMeans(n_clusters=500, random_state=random_seed)

for chunk in np.array_split(cluster_data,3):
    clusterer.partial_fit(chunk)
search = cluster_data.loc[10].reshape(1, -1)
print clusterer.predict(search)
db_client.close()

labels = clusterer.labels_
print len(labels)
print labels
print labels[10]

for i,label in enumerate(labels):
    if label==5:
        print data.ix[i]._url