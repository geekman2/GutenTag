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

    def parse_page(self, response):
        titles = response.xpath('//a[@class="stitle"]')

        for title in titles:
            item = MonkPageItem()
            title_text = title.xpath('text()').extract()
            title_url = title.xpath('@href').extract()
            item['title'] = title_text
            item['url'] = title_url
            request = scrapy.Request(url=title_url,
                                     callback=self.parse_page2())
            yield item

    def parse_page2(self, response):
        item = MonkStoryItem
        text = response.xpath('//*[@id="storytext"/text()]').extract()


