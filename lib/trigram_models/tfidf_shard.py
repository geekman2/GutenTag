#-------------------------------------------------------------------------------
# Name:         module1
# Purpose:
# Author:       Devon Muraoka
# Created:      
# Copyright:   (c) Devon Muraoka, Bharat Ramanathan 
#-------------------------------------------------------------------------------
from lib.trigram_models import CorpusModel, SemanticModels, SimilarityModel
import logging
import settings
import os
import glob
import numpy as np
import datashader
import gensim
import csv

cwd = settings.project_root
working_directory = os.path.join(cwd,'tmp','modeldir')
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

logger.info(working_directory)
corpus_model = CorpusModel()
corpus = corpus_model.load_corpus()
dictionary = corpus_model.load_dict()
tfidf_corpus = corpus_model.load_tfidf_corpus()

shard_dir = os.path.join(cwd, 'tmp', 'tfidf_fragments')
if not os.path.exists(shard_dir):
    os.mkdir(shard_dir)
    tmp = []
    chunk_size = 100000
    for i, row in enumerate(tfidf_corpus):
        tmp.append(row)
        if i % chunk_size == 0:
            filename = os.path.join(cwd, 'tmp', 'tfidf_fragments', 'tfidf.part{}.csv'.format(i/chunk_size))
            with open(filename, 'wb') as f:
                logger.info('Writing part {} to disk'.format(i/chunk_size))
                writer = csv.writer(f)
                writer.writerows(tmp)
            tmp = []

#all_files = glob.glob(os.path.join(shard_dir, "*.csv"))
#df = pd.concat(pd.read_csv(f) for f in all_files)
