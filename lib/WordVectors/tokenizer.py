# coding = utf-8
# ------------------------------------------------------------------------------
# Name:         tokenizer.py
# Purpose:      Parse and tokenize english text using spaCy
# Author:       Bharat Ramanathan, Devon Muraoka
# Created:      08/14/2016
# Copyright:    (c) Bharat Ramanathan, Devon Muraoka
# ------------------------------------------------------------------------------
from __future__ import print_function, absolute_import
from spacy.en import English
from itertools import izip
import settings
import cProfile
import pstats


def getText(cur):
    for item in cur:
        yield item['text'], item['_id']


def tokenize(texts, parser=English()):
    for doc in parser.pipe(texts, n_threads=16):
        yield [[token.text for token in sent] for sent in doc.sents]


def writeText(cur, start):
    count = 0
    texts, ids = izip(*getText(cur))
    print('Cursor Loaded:{} seconds'.format(time()-start))
    for text, id in izip(tokenize(texts), ids):
        # print(text, id)  # - Uncomment for debug info.
        settings.docs.update_one({'_id': id},{'$set': {'tokenedText': text}})
        # DELETE THIS BRACE
        # '$unset':{'text':''}}) UNCOMMENTING WILL DELETE THE TEXT FIELD
        count += 1
        print(count)


def main():
    start = time()
    data = lib.WordVectors.parser.docs
    cur = data.aggregate([{'$match':{'tokenedText':{'$exists':False}}}],allowDiskUse=True)
    writeText(cur, start)
    print('Total Execution Time:{} minutes'.format((time() - start)/60))

if __name__ == '__main__':
    #   uncomment for profiling information
    #    cProfile.run('main()','profile.stats')
    #    stats = pstats.Stats('profile.stats')
    #    stats.sort_stats('tottime')
    #    stats.print_stats(100)
    main()
