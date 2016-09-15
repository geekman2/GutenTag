import os
import gensim
from time import time
import logging

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class TopicModels(object):
    def __init__(self, tmp_folder, corpus, dictionary):
        self.tmp_folder = tmp_folder
        self.corpus = corpus
        self.dictionary = dictionary

    def build_lda_model(self, n_topics=10, chunks=3000, n_passes=3, n_jobs=3):
        lda_file = os.path.join(self.tmp_folder, 'lda_model')
        if os.path.isfile(lda_file):
            lda_model = gensim.models.LdaModel.load(lda_file)
        else:
            params = {'corpus': self.corpus,
                      'id2word': self.dictionary,
                      'num_topics': n_topics,
                      'chunksize': chunks,
                      'workers': n_jobs,
                      'passes': n_passes,
                      }
            lda_model = gensim.models.LdaMulticore(**params)
            lda_model.save(lda_file)
        return lda_model

    def build_lda_corpus(self, n_topics=10, passes=3, n_jobs=3):
        lda_file = os.path.join(self.tmp_folder, 'lda_corpus.mm')
        if os.path.isfile(lda_file):
            lda_corpus = gensim.corpora.mmcorpus.MmCorpus(lda_file)
        else:
            lda_model = self.build_lda_model(n_topics=n_topics,
                                             n_passes=passes,
                                             n_jobs=n_jobs)
            lda_corpus = lda_model[self.corpus]
            gensim.corpora.MmCorpus.serialize(lda_file, lda_corpus)
        return lda_corpus

    def build_hdp_model(self, max_topics=150, chunks=3000):
        hdp_file = os.path.join(self.tmp_folder, 'hdp_model')
        if os.path.isfile(hdp_file):
            hdp_model = gensim.models.HdpModel.load(hdp_file)
        else:
            params = {'corpus': self.corpus,
                      'id2word': self.dictionary,
                      'T': max_topics,
                      'chunksize': chunks
                      }
            hdp_model = gensim.models.HdpModel(**params)
            hdp_model.save(hdp_file)
        return hdp_model

    def build_hdp_corpus(self, max_topics=150, chunks=3000):
        hdp_file = os.path.join(self.tmp_folder, 'hdp_corpus')
        if os.path.isfile(hdp_file):
            hdp_corpus = gensim.corpora.mmcorpus.MmCorpus(hdp_file)
        else:
            hdp_model = self.build_hdp_model(max_topics=max_topics,
                                             chunks=chunks
                                             )
            hdp_corpus = hdp_model[self.corpus]
            gensim.corpora.MmCorpus.serialize(hdp_file, hdp_corpus)
        return hdp_corpus

if __name__ == '__main__':
    from lib.topicModels.vector_models import VectorModels

    start_corpus = time()
    cwd = os.getcwd()
    data_folder = os.path.join(cwd, 'tmp', 'testFiles', '*')
    tmp_folder = os.path.join(cwd, 'tmp', 'modeldir')

    vectors = VectorModels(data_folder, tmp_folder)
    corpus, dictionary = vectors.load_corpus()
    tfidf_corpus = vectors.build_tfidf_corpus(corpus, dictionary)

    semantic = TopicModels(tmp_folder, tfidf_corpus, dictionary)
    semantic.build_hdp_corpus()

    stop_corpus = time()
    corpus_time = round(stop_corpus - start_corpus, 3)
    logger.info('time taken to build corpus = {}s'.format(corpus_time))
