# -------------------------------------------------------------------------------
# Name:         module1
# Purpose:
# Author:       Devon Muraoka
# Created:      7/13/2016
# Copyright:    (c) Devon Muraoka
# -------------------------------------------------------------------------------

from __future__ import absolute_import
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import scrapy
import logging
import datetime
from Monk import items
import logging


# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# file_handler = logging.FileHandler("crawl.log")
# formatter = logging.Formatter()


class FanfictionSpider(CrawlSpider):
    name = "fanfiction_spider"
    allowed_domains = ["fanfiction.net"]
    start_urls = [
        "http://www.fanfiction.net/anime/Seto-no-Hanayome/"
    ]
    rules = [
        Rule(LinkExtractor(allow=r'\/[a-z]*\/.*\?&srt=1&r=103&p=[0-9]'),
             callback="parse_page", follow=True)
    ]
    logging.basicConfig(filename=datetime.datetime.now().strftime('crawl_%H_%M_%d_%m_%Y.log'),
                        format='[%(asctime)s]%(levelname)s: %(message)s',
                        level=logging.ERROR)

    def parse_page(self, response):
        """

        :rtype: object
        :param response: 
        """
        logging.info("Page Parse started")
        titles = response.xpath('//a[@class="stitle"]')
        for title in titles:
            item = items.MonkPageItem()
            title_text = title.xpath('text()').extract()[0]
            title_url = response.urljoin(title.xpath('@href').extract()[0])
            logging.info("Current URL:{}".format(title_url))
            item['title'] = title_text
            item['url'] = title_url
            request = scrapy.Request(url=title_url,
                                     callback=self.parse_story_page)
            request.meta['item'] = item
            yield request


    def parse_story_page(self, response):
        """
        
        :param response: 
        :rtype: MonkStoryItem
        """
        item = items.MonkStoryItem()
        n_grams = {}
        n_gram_frequencies = {}
        text = "".join(response.xpath('//*[@id="storytext"]//text()').extract()).strip()
        logging.debug("Length of text:{}".format(len(text)))
        for ngram in self.generate_ngrams(text, 3):
            if ngram in n_grams:
                n_grams[ngram] += 1
            else:
                n_grams[ngram] = 1
        for key in n_grams:
            n_gram_frequencies[key] = float(n_grams[key]) / len(text)
        item["trigram_frequency"] = n_gram_frequencies  # n_grams
        yield item

    @staticmethod
    def generate_ngrams(text, n):
        for i in xrange(len(text) - n):
            yield text[i:i + n]
