# -------------------------------------------------------------------------------
# Name:         module1
# Purpose:
# Author:       Devon Muraoka
# Created:      7/13/2016
# Copyright:    (c) Devon Muraoka
# -------------------------------------------------------------------------------

from scrapy import CrawlSpider
from bs4 import BeautifulSoup as beauty


class FanfictionSpider(CrawlSpider):
    name = "fanfiction_spider"
    allowed_domains = ["google.com"]
    start_urls = [
        "http://www.fanfiction.net/anime/Seto-no-Hanayome/"

    ]

    def parse(self, response):
        titles = response.xpath('//a[@class="stitle"]')

        for title in titles:
            item = MonkPageItem()
            item['title'] = title.xpath('text()').extract()
            item['url'] = sel.xpath('@href').extract()
            yield item




