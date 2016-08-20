# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import pymongo
from scrapy.conf import settings


class MonkPipeline(object):
    def __init__(self):
        connection = pymongo.MongoClient("mongodb://{}:{}".format(
            settings['MONGODB_SERVER'],
            settings['MONGODB_PORT'])
        )
        db = connection[settings['MONGODB_DB']]
        self.collection = db[settings['MONGODB_COLLECTION']]

    def process_item(self, item, spider):
        if item["trigram_frequency"] and item["fandom"] != "":
            item["url"] = item["url"].replace("/m.","/www.")
            self.collection.insert(dict(item))
        return item
