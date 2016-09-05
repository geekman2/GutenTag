from itertools import izip
import spacy
from lib.WordVectors.parser import docs

parser = spacy.load('en')

doc = docs.find_one({'text':{'$exists':1}},{'text':1, '_id':0})
parsed = parser(doc['text'])

def pre_process(texts,stops=True,lemmatize=True,stemm=True)