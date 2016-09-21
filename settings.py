#-------------------------------------------------------------------------------
# Name:         GutenTag Settings
# Purpose:      Set parameters for the GutenTag framework
# Author:       Devon Muraoka
# Created:      9/5/16
# Copyright:   (c) Devon Muraoka, Bharat Ramanathan 
#-------------------------------------------------------------------------------
import pymongo
import os

debug = True
# Necessary connection variables.

# db_ip = '159.203.187.28' remote IP, deprecated
db_ip = 'localhost'
db_port = '27017'
db = pymongo.MongoClient('mongodb://{}:{}'.format(db_ip, db_port))
docs = db.data.fiction

# Root Directory
project_root = os.path.dirname(os.path.realpath(__file__))  # GutenTag Project root