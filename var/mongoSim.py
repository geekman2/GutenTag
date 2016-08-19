# coding: utf-8
from __future__ import print_function, absolute_import
# from pymongo import MongoClient
from glob import glob
# from WordVectors.tokenizer import tokenize, getText
import json
import codecs

from itertools import islice  # , izip

"""
def writeText(cur):
    texts, ids = izip(*getText(cur))
    with open('./var/data.json', 'a') as f:
        for text, ids in izip(tokenize(texts), ids):
            data = {u'tokenedText': text, '_id': ids}
            json.dump(data, f)
            f.write('\n')


def getCur():
    db_ip = '159.203.187.28'
    db_port = '27017'
    db = MongoClient('mongodb://{}:{}'.format(db_ip, db_port))
    docs = db.data.fiction
    return docs.find({'text': {'$exists': 'true'}}, {'text': 1})


def writeFiles(cur):
    for item in cur:
        f = codecs.open('./var/testFiles/{}'.format(item['_id']), mode='w',
                        encoding='utf-8')
        f.write(item['text'])
        f.close()

"""


def getData(fileLoc):
    if fileLoc is None:
        fileLoc = '/home/dante/Documents/GutenTag/var/testFiles/'
    for item in glob(fileLoc+'*'):
        f = codecs.open(item, encoding='utf-8', mode='r')
        filedat = u""
        for line in f:
            filedat += line
        f.close()
        yield {'text': filedat,
               '_id': item.split('/')[-1]}


def simMongoDb(n=10, array=True, dataLoc=None, jsonLoc=None):
    if not array:
        for item in islice(getData(dataLoc), n):
            yield item
    else:
        for item in islice(readJson(jsonLoc), n):
            yield item


def readJson(fileLoc):
    if fileLoc is None:
        fileLoc = '/home/dante/Documents/GutenTag/tmp/bowdata.json'
    with open(fileLoc, 'r') as f:
        for line in f:
            yield json.loads(line)

if __name__ == '__main__':
    # Uncomment these likes if you want to load the data for the simulator
    # cur = getCur()
    # writeFiles(cur[:10000])
    cur = simMongoDb(n=10, array=True)
    # writeText(cur)
    for item in cur:
        print(item['tokenedText'])
