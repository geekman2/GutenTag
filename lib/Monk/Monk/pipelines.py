# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import pymongo
from scrapy.conf import settings
import logging


class MonkPipeline(object):
    def __init__(self):
        connection = pymongo.MongoClient("mongodb://{}:{}".format(
            settings['MONGODB_SERVER'],
            settings['MONGODB_PORT'])
        )
        db = connection[settings['MONGODB_DB']]
        self.collection = db[settings['MONGODB_COLLECTION']]

    def process_item(self, item, spider):
        if item["fandom"] != "":
            item_dict = dict(item)
            item_dict['genres'] = self.parse_genre(item['genres'])
            item_dict['trigram_frequency'] = self.generate_ngram_frequencies(item['text'],3)
            item_dict["url"] = item["url"].replace("/m.","/www.")
            self.collection.insert(item_dict)
        return item

    @staticmethod
    def generate_ngrams(text, n):
        text = text.replace(".", "*")
        text = text.replace("$", "^")
        for i in xrange(len(text) - n):
            yield text[i:i + n]

    def generate_ngram_frequencies(self, text, n):
        n_grams = {}
        n_gram_frequencies = {}
        for ngram in self.generate_ngrams(text, n):
            if ngram in n_grams:
                n_grams[ngram] += 1
            else:
                n_grams[ngram] = 1
        for key in n_grams:
            n_gram_frequencies[key] = float(n_grams[key]) / len(text)
        return n_gram_frequencies

    @staticmethod
    def parse_genre(lst):
        valid_genres = ["Adventure", "Angst", "Crime", "Drama", "Family", "Fantasy", "Friendship", "General", "Horror",
                        "Humor",
                        "Hurt/Comfort", "Mystery", "Parody", "Poetry", "Romance", "Sci-Fi", "Spiritual", "Supernatural",
                        "Suspense", "Tragedy", "Western"]
        for i in lst:
            if "Rated:" in i:
                genre = i.split(",")[2].strip()
                logging.debug('STRIPPED TEXT:{}'.format(genre.strip()))
                if any(word in genre for word in valid_genres):
                    if "&" in genre:
                        return genre.split(" & ")
                    else:
                        return [genre]


