# coding = utf-8
# ------------------------------------------------------------------------------
# Name:         parser.py
# Purpose:      Parse and tokenize english text using spaCy
# Author:       Bharat Ramanathan
# Created:      08/13/2016
# Copyright:    (c) Bharat Ramanathan
# ------------------------------------------------------------------------------
from spacy.en import English


def tokenize(doc, parser=English()):
    # Intialize the parser and tokenize the text retrieved
    return parser(doc)
