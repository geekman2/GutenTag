# coding = utf-8
# ------------------------------------------------------------------------------
# Name:         tokenizer.py
# Purpose:      Parse and tokenize english text using spaCy
# Author:       Bharat Ramanathan
# Created:      08/14/2016
# Copyright:    (c) Bharat Ramanathan
# ------------------------------------------------------------------------------
from __future__ import print_function, absolute_import
import spacy
import multiprocessing
import WordVectors.parser


def tokenize(texts, parser=spacy.en.English()):
    # Intialize the parser and tokenize the text retrieved
    for doc in parser.pipe(texts, n_threads=16):
        sentences = []
        for span in doc.sents:
            sent = [token.text for token in span]
            sentences.append(sent)
        yield sentences


def worker(doc):
    # Get the parsed data and update the document with tokenedText
    parsed = tokenize(doc['text'])
    for sentence in parsed:
        print(sentence)
    # WordVectors.parser.docs.update_one({'_id': doc['_id']},
    #                                   {'$set': {'tokenedText': parsed}, })
# DELETE THIS BRACE
# '$unset':{'text':''}}) UNCOMMENTING WILL DELETE THE TEXT FIELD


def tokenizeMultiple():
    # Multiprocessing-fu with the tokenization
    cur = WordVectors.parser.docs.find({'text': {'$exists': 'true'}},
                                       {'text': 1})
    pool = multiprocessing.Pool(8)
    pool.map_async(worker, cur[:100])
    pool.close()
    pool.join()

if __name__ == '__main__':
    tokenizeMultiple()
