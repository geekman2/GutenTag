# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class MonkPageItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    title = scrapy.Field()
    url = scrapy.Field()


class MonkStoryItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    trigram_frequency = scrapy.Field()
