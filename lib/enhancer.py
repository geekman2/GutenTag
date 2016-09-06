#-------------------------------------------------------------------------------
# Name:         Enhancer
# Purpose:      Check database for entries without a given attribute,
#               and if they have no text, pull text from the URL
# Author:       Devon Muraoka
# Created:      9/5/16
# Copyright:    (c) Devon Muraoka, Bharat Ramanathan
#-------------------------------------------------------------------------------
import pymongo
import bs4
import requests
import var.settings as settings

class Enhancer(object):


    def __init__(self,db):
        self.db = db

    def pull(self,field):
        cursor = self.db.find({field:{'exists':'false'}})
        cursor.batch_size(1000)
        for doc in cursor:
            url = doc.url
            page = requests.get(url)
            soup = bs4.BeautifulSoup(page.text)





if __name__ == "__main__":
    db = settings.db
    docs = settings.docs
    puller = Enhancer(docs)
    puller.pull("text")