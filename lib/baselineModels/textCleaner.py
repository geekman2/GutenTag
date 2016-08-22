from __future__ import print_function, absolute_import
import re
from itertools import izip
from nltk.corpus import stopwords
from var.mongoSim import simMongoDb
from spacy.en import English
from os import getcwd
import json


class cleanText(object):

    def __init__(self, cur):
        self.cur = cur
        self.texts = None
        self.id = None

    def removePunct(self):
        for text in self.texts:
            onlyText = re.sub("[^a-zA-Z]",  # The pattern to search for
                              " ",  # The pattern to replace it with
                              text)  # The text to search
            yield onlyText

    def tokenize(self, parser=English()):
        stops = set(stopwords.words("english"))
        for doc in parser.pipe(self.removePunct(), n_threads=16):
                yield [token.text.lower() for token in doc
                       if token.text.lower() not in stops]

    def getText(self):
        for item in self.cur:
            yield item['text'], item['_id']

    def __iter__(self):
        self.texts, self.ids = izip(*self.getText())
        for item, id in izip(self.tokenize(), self.ids):
            yield {'text': re.sub(' +', ' ', u' '.join(item)), '_id': id}


def writeCleanText(cur, outFile):
    with open(outFile, 'a') as f:
        for item in cleanText(cur):
            json.dump(item, f)
            f.write('\n')


if __name__ == '__main__':
    cur = simMongoDb(n=10000, array=False, )
    dataPath = "{}/var/".format(getcwd())
    dataFile = dataPath+"bowdata.json"
    writeCleanText(cur, dataFile)
