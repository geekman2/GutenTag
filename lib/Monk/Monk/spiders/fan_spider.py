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


class FanfictionSpider(CrawlSpider):
    name = "fanfiction_spider"
    allowed_domains = ["fanfiction.net"]
    start_urls = [
        "https://m.fanfiction.net/book",
        "https://m.fanfiction.net/book/Percy-Jackson-and-the-Olympians/",
        "https://m.fanfiction.net/tv/Chuck/",
        "https://m.fanfiction.net/anime/Fullmetal-Alchemist/"
    ]
    rules = [
        Rule(LinkExtractor(allow=[r'https://m.fanfiction.net/s/12011067/1/Hunting-a-Lost-Shadow'],
                           deny=[r'communities', r'.php', r'/r/', r'/u/', r'/forum/', r'crossover']),
             callback="parse_page", follow=True),
    ]

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
        pages = response.xpath('//a[contains(@href,"/s/")]')
        for page in pages:
            item = items.MonkStoryItem()
            title_url = response.urljoin(page.xpath('@href').extract()[0])
            item['url'] = title_url
            request = scrapy.Request(url=title_url,
                                     callback=self.parse_story_page)
            request.meta['item'] = item
            yield request

        text = "".join(response.xpath('//*[@id="storycontent"]//text()').extract()).strip()
        fandom = response.xpath('//*[@id="content"]/a[2]//text()').extract()
        if type(fandom) == list:
            fandom = "".join(fandom)
        info = response.xpath('//*[@id="content"]//text()').extract()
        item["url"] = response.url
        item["fandom"] = fandom
        item["genres"] = info
        item["text"] = text
        yield item


