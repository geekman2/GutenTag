# -------------------------------------------------------------------------------
# Name:         Enhancer
# Purpose:      Check database for entries without a given attribute,
#               and if they have no text, pull text from the URL
# Author:       Devon Muraoka
# Created:      9/5/16
# Copyright:    (c) Devon Muraoka, Bharat Ramanathan
# -------------------------------------------------------------------------------
import pymongo
import time
import bs4
import requests
import logging
import var.settings as settings

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class Enhancer(object):
    def __init__(self, db):
        self.db = db

    def pull(self, field):
        cursor = self.db.find({"text": {'$exists': False}})
        count = 0
        for doc in cursor:
            if count % 1000 == 0:
                time.sleep(5)
            url = doc['url']
            _id = doc['_id']
            url = url.replace("/www", "/m")
            page = requests.get(url)
            soup = bs4.BeautifulSoup(page.text, 'lxml')
            # The code that follows is hard coded and sub-optimal
            # TODO optimize this code
            # This code will only pull text and only from fanfiction.net
            text = " ".join([x.text for x in soup.find_all('p')])
            self.db.update_one({'_id': _id}, {'$set': {'text': text}})
            count += 1
            logging.info(count, url)


if __name__ == "__main__":
    db = settings.db
    docs = settings.docs
    puller = Enhancer(docs)
    puller.pull('text')
