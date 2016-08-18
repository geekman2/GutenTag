from gensim.models import Doc2Vec
from pprint import pprint
import os

ids = ['57ac0428058acb4a68a52a47',
       '57ac042a058acb4a68a52a4d',
       '57ac042b058acb4a68a52a4e',
       '57ac042d058acb4a68a52a55',
       '57ac042e058acb4a68a52a59',
       '57ac042f058acb4a68a52a5b',
       '57ac042f058acb4a68a52a5c',
       '57ac042f058acb4a68a52a5e',
       '57ac0430058acb4a68a52a5f']

modelPath = "{}/var/".format(os.getcwd())
modelFile = modelPath + "trial.model"
model = Doc2Vec.load(modelFile)

pprint(model.docvecs.most_similar('57ac042f058acb4a68a52a5b'))
pprint(model.most_similar('good'))
