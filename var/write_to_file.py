# -------------------------------------------------------------------------------
# Name:         module1
# Purpose:
# Author:       Devon Muraoka
# Created:      
# Copyright:   (c) Devon Muraoka, Bharat Ramanathan 
# -------------------------------------------------------------------------------
from __future__ import absolute_import, print_function

import io
import logging
import os

import settings as settings

logger = logging.getLogger('text_writing')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def write_files(cur,output_path):
    count = 0
    for item in cur:
        with io.open('{}/{}.txt'.format(output_path,item['_id']),mode='w',encoding='utf-8') as working_file:
            working_file.write(item['text'])
        count += 1
        if count % 10000 == 0:
            if count > 1000000:
                logger.info('Written {} million documents to text files'.format(count/1000000000.0))
            else:
                logger.info('Written {} thousand documents to text files'.format(count/1000))

if __name__ == "__main__":
    output_path = os.path.join(os.getcwd(),'test_files')
    if not os.path.exists(output_path):
        print(output_path)
        os.mkdir(output_path)
    cursor = settings.docs.find({'text': {'$exists': 'true'}}, {'text': 1})
    write_files(cursor,output_path)