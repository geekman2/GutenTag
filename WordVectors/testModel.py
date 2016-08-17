from gensim.models import Doc2Vec


ids = ['57abc460058acb2a38e18c5c', '57abd8cd058acb4a68a49705',
       '57abd147058acb4a68a47b5c', '57abd504058acb4a68a48809',
       '57abd419058acb4a68a484d4', '57abd558058acb4a68a4893b']

model = Doc2Vec.load('/home/dante/Documents/GutenTag/var/trial.model')

print model.docvecs.most_similar('57abd8cd058acb4a68a49705')
print model.most_similar('good')
