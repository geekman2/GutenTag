import lib.WordVectors.parser as mongoClient
from collections import defaultdict
from pprint import pprint
import numpy as np
import time

start = time.time()
df = defaultdict(int)
docs = mongoClient.docs
cur = docs.find({'trigram_frequency': {'$exists': 1}}, {'trigram_frequency':1, '_id':0})
for doc in cur[:10000]:
    for term in doc['trigram_frequency'].keys():
        df[term] += 1

pprint(time.time()-start)

"""
tf = cur['trigram_frequency']
total_keys = len(tf.keys())
factor = total_keys/sum(tf.values())
freqs = np.array(tf.values())
print(factor*freqs)
"""
