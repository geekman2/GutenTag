from __future__ import print_function, absolute_import
import itertools
import nltk
import spacy
import lib.WordVectors.parser as mongoClient
import time
import gensim


class cleanText(object):

    def __init__(self, docs, n_docs=None, lemmatize_it=True, stem_it=True, normalize_it=True):
        self.docs = docs
        self.n_docs = n_docs
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
            if token.pos_ != 'PROPN' and token.ent_type_ not in ['PERSON', 'ORG']:
                return token.orth_
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

    def pre_process(self, texts):
        for doc in self.parser.pipe(texts, n_threads=4):
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

    def get_texts(self):
        cur = self.docs.find({'text': {'$exists': 'true'}}, {'text': 1})
        for item in cur[:self.n_docs]:
            yield item['text']
        if not self.n_docs:
            for item in cur:
                yield item

    def __iter__(self):
        texts = self.get_texts()
        for item in self.pre_process(texts):
            yield item

if __name__ == '__main__':
    start = time.time()
    docs = mongoClient.docs
    cleaned = cleanText(docs=docs, n_docs=10)
    for item in cleaned:
        print(item)
    for item in cleaned:
        print(item)

    print(time.time() - start)
