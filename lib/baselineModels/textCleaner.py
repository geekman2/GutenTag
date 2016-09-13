from __future__ import print_function, absolute_import
import itertools
import nltk
import spacy
import lib.WordVectors.parser as mongoClient
import time
import gensim


class TextProcessor(object):
    def __init__(self, lemmatize_it=True, stem_it=True, normalize_it=True):
        self.lemmatize_it = lemmatize_it
        self.stem_it = stem_it
        self.normalize_it = normalize_it
        self.parser = spacy.load('en')
        self.stemmer = gensim.parsing.PorterStemmer()
        self.stops = set(nltk.corpus.stopwords.words('english'))

    def stem(self, token):
        if self.stem_it:
            return self.stemmer.stem(token)
        else:
            return token

    def lemmatizer(self, token):
        if self.lemmatize_it:
            if token.pos_ != 'PROPN' and token.ent_type_ not in {'PERSON', 'ORG'}:
                return token.lemma_
            else:
                return 'ENTITY'
        else:
            return token.orth_

    def normalize(self,token):
        if self.normalize_it:
            if token.lower() not in self.stops:
                return token
        else:
            return token

    def pre_processor(self, docs):
        for doc in self.parser.pipe(docs, n_threads=4):
            processed = []
            for sent in doc.sents:
                for token in sent:
                    lemmatized = self.lemmatizer(token)
                    normalized = self.normalize(lemmatized)
                    if normalized:
                        stemmed = self.stem(normalized)
                        processed.append(stemmed)
                    else:
                        continue
            yield processed

if __name__ == '__main__':
    start = time.time()
    docs = mongoClient.docs
    cur = mongoClient.docs.find({'text': {'$exists': 'true'}}, {'text': 1})
    cleaned = TextProcessor()
    def temp(cur):
        for item in cur[:100]:
            yield item['text']
    for item in cleaned.pre_processor(temp(cur)): pass
    print(time.time() - start)