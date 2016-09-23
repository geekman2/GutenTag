import os
import cPickle as pickle
import csv
import settings
from nltk.tokenize import RegexpTokenizer

tmp_dir = os.path.join(settings.project_root, 'tmp')

map_file = os.path.join(tmp_dir, 'genre_mapping')
with open(map_file, 'rb') as map_f:
    mapping = pickle.load(map_f)

sample_dir = os.path.join(tmp_dir, 'sample_dir')
csv_file = os.path.join(tmp_dir, 'data_file.csv')

tokenizer = RegexpTokenizer(r'\w+')

with open(csv_file, 'w+') as csv_f:
    field_names = ['genre', 'text']
    writer = csv.DictWriter(csv_f, fieldnames=field_names)
    genre_list = {u'Humor', u'Sci-Fi', u'Family', u'Romance', u'Supernatural'}
    for key, values in mapping.iteritems():
        if key in genre_list:
            for value in values:
                with open(os.path.join(sample_dir, value), 'r') as f:
                    tokened = tokenizer.tokenize(f.read())
                    writer.writerow({'genre': key, 'text': " ".join(tokened)})
