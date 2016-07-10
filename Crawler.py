# -------------------------------------------------------------------------------
# Name:        module1
# Purpose:
# Author:      Devon Muraoka
# Created:     7/8/2016
# Copyright:   (c) Devon Muraoka 2016
# -------------------------------------------------------------------------------
import requests
import simplejson


class Crawler:
    def __init__(self, seed_url):
        self.crawled = set()
        self.to_crawl = set()
        self.seed_url = seed_url
        self.page_raw = requests.get(self.seed_url).text

    @staticmethod
    def get_next_target(page):
        """
        :rtype: string, int
        :param page:string, a webpage in raw HTML
        :return : link_text:the text of the first link it finds
        link_end: the index where link_text ends
        """
        #TODO fix link gathering so that both absolute and relative links are scraped correctly
        link_start = page.find("href=")
        if link_start > -1:
            link_end = page.find('">', link_start + 7)
            link_text = page[link_start + 5:link_end]
            return link_text, link_end
        else:
            return None, None

    def get_all_links(self, page):
        """
        :rtype: list
        :param page: string, a webpage in raw HTML
        :return : links: a list of links contained on the input page
        """
        links = []
        url, end_position = self.get_next_target(page)
        while url:
            if self.is_story(url) or self.is_page_link(url):
                links.append(url)
            page = page[end_position:]
            url, end_position = self.get_next_target(page)
        return links

    @staticmethod
    def is_story(url):
        """
        Check the url of a given story against a set of predefined patterns
        if it matches one of the patterns, return true, otherwise, return false
        """
        if "fanfiction.com" in url and "/s/" in url:
            return True
        else:
            return False

    @staticmethod
    def is_page_link(url):
        """
        Check the url against a set of predefined patterns for page links
        if it matches one of the patterns, return true, otherwise, return
        """
        if "&p=" in url:
            return True
        else:
            return False

    def start_crawl(self):
        #Add a starting set of pages to the to_crawl list
        self.to_crawl.update(self.get_all_links(self.page_raw))
        for url_to_crawl in self.to_crawl:
            page_to_crawl = requests.get(url_to_crawl).text
            self.to_crawl.update(self.get_all_links(page_to_crawl))
            self.crawled.add(url_to_crawl)

Crawler("https://www.fanfiction.net/anime/Seto-no-Hanayome/").start_crawl()