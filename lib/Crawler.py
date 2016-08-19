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
    def __init__(self, domain, seed_page):
        self.crawled = set()
        self.to_crawl = []
        self.absolute_prepend = "https://www."
        self.domain = domain
        self.seed_page = seed_page
        self.seed_url = "{protocol}{domain}/{page}".format(protocol=self.absolute_prepend,
                                                           domain=self.domain,
                                                           page=seed_page)
        self.page_raw = self.get_body(self.seed_url)
        self.domain_criteria = {
            "fanfiction.net/anime/Seto-no-Hanayome": "/s/"
        }

    @staticmethod
    def get_body(url):
        page = requests.get(url).text
        head_end = page.find("/head")
        body = page[head_end:]
        body = body.replace("'", "\"")
        return body

    @staticmethod
    def get_next_target(page):
        """
        :rtype: string, int
        :param page:string, a webpage in raw HTML
        :return : link_text:the text of the first link it finds
        link_end: the index where link_text ends
        """
        # TODO fix link gathering so that both absolute and relative links are scraped correctly
        link_start = page.find("href=")
        if link_start > -1:
            link_end = page.find('"', link_start + 7)
            link_text = page[link_start + 6:link_end]
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
            if self.is_valid_link(url) and url not in self.crawled:
                link = self.absolute_prepend + self.domain + url
                links.append(link)
            page = page[end_position:]
            url, end_position = self.get_next_target(page)
        return links

    def is_valid_link(self, url):
        """
        Check the url against a set of predefined patterns for page links
        if it matches one of the patterns, return true, otherwise, return
        :param url:
        """
        criterion = self.domain_criteria[self.domain]
        if url.startswith(criterion):
            return True
        else:
            return False

    def start_crawl(self):
        # Add a starting set of pages to the to_crawl list
        self.to_crawl.extend(self.get_all_links(self.page_raw))
        for url_to_crawl in self.to_crawl:
            page_to_crawl = self.get_body(url_to_crawl)
            self.to_crawl.extend(self.get_all_links(page_to_crawl))
            print self.crawled
            self.crawled.add(url_to_crawl)


Crawler("fanfiction.net/anime/Seto-no-Hanayome", "").start_crawl()
