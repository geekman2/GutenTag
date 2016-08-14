# coding = utf-8
# ------------------------------------------------------------------------------
# Name:         parser.py
# Purpose:      Parse and tokenize english text using spaCy
# Author:       Bharat Ramanathan
# Created:      08/13/2016
# Copyright:    (c) Bharat Ramanathan
# ------------------------------------------------------------------------------
from __future__ import print_function
# from spacy.en import English
from nltk import wordpunct_tokenize
import tokenizer


def tokenize(doc, parser=wordpunct_tokenize):
    # Intialize the parser and tokenize the text retrieved
    return parser(doc.lower())


cur = tokenizer.getCursor()
tokenedDocs = []
for item in cur[:1000]:
    tokenedDocs.append(tokenize(tokenizer.getText(item)))
print(len(tokenedDocs))
