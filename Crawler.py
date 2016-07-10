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
        self.crawled = {}
        self.seed_page = seed_url
        self.page_raw = requests.get(self.seed_page).text

    @staticmethod
    def get_next_target(page):
        """

        :rtype: string, int
        :param page:string, a webpage in raw HTML
        :return : link_text:the text of the first link it finds
        link_end: the index where link_text ends
        """
        link_start = page.find("href=")
        if link_start > -1:
            link_end = page.find("\"", link_start + 7)
            link_text = page[link_start + 6:]
            return link_text, link_end

    def get_all_links(self, page):
        links = []
        url, end_position = self.get_next_target(page)
        while url:
            links.append(url)
            page = page[end_position:]
            url, end_position = self.get_next_target(page)
        return links

    def is_story(self, url):
        pass
