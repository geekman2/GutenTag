# coding = utf-8
# ------------------------------------------------------------------------------
# Name:         Create trigram docs
# Purpose:      Create trigram docs #TODO improve this description
# Author:       Bharat Ramanathan, Devon Muraoka
# Created:      9/6/2016
# Copyright:    (c) Bharat Ramanathan, Devon Muraoka
# ------------------------------------------------------------------------------
from itertools import izip
import spacy
import var.settings as settings

parser = spacy.load('en')

doc = settings.docs.find_one({'text':{'$exists':1}},{'text':1, '_id':0})
parsed = parser(doc['text'])

def pre_process(texts,stops=True,lemmatize=True,stemm=True):
    pass