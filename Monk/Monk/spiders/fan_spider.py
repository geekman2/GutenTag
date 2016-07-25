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
        "http://m.fanfiction.net/"
    ]
    rules = [
        Rule(LinkExtractor(allow=[r'.*'],
                           deny=[r'communities', r'.php', r'/r/', r'/u/',r'/forum/']),
             callback="parse_story_page", follow=True),
    ]
    logging.basicConfig(filename=datetime.datetime.now().strftime('crawl_%H_%M_%d_%m_%Y.log'),
                        format='[%(asctime)s]%(levelname)s: %(message)s')

    def parse_page(self, response):
        """

        :rtype: object
        :param response: 
        """
        logging.debug("Page Parse started on:{}".format(response.url))
        # titles = response.xpath('//a[@class="stitle"]')   Desktop site variant
        titles = response.xpath('//a[contains(@href,"/s/")]')  # Mobile site variant
        for title in titles:
            item = items.MonkPageItem()
            title_text = title.xpath('text()').extract()[0]
            title_url = response.urljoin(title.xpath('@href').extract()[0])
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
        logging.debug("Story Parse Started on:{}".format(response.url))
        item = items.MonkStoryItem()
        n_grams = {}
        n_gram_frequencies = {}
        # text = "".join(response.xpath('//*[@id="storytext"]//text()').extract()).strip()
        text = "".join(response.xpath('//*[@id="storycontent"]//text()').extract()).strip()
        for ngram in self.generate_ngrams(text, 3):
            if ngram in n_grams:
                n_grams[ngram] += 1
            else:
                n_grams[ngram] = 1
        for key in n_grams:
            n_gram_frequencies[key] = float(n_grams[key]) / len(text)
        item["url"] = response.url
        item["trigram_frequency"] = n_gram_frequencies  # n_grams
        yield item

    @staticmethod
    def generate_ngrams(text, n):
        for i in xrange(len(text) - n):
            yield text[i:i + n]
