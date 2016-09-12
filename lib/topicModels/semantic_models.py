import os
import gensim


class TopicModels(object):
    def __init__(self, tmp_folder, corpus, dictionary):
        self.tmp_folder = tmp_folder
        self.corpus = corpus
        self.dictionary = dictionary

    def build_lda_model(self, n_topics=100, chunks=3000, n_passes=3, n_jobs=3):
        lda_file = os.path.join(self.tmp_folder, 'lda_model')
        if os.path.isfile(lda_file):
            lda_model = gensim.models.LdaModel.load(lda_file)
        else:
            params = {'corpus': self.corpus,
                      'id2word': self.dictionary,
                      'num_topics': n_topics,
                      'chunksize': chunks,
                      'workers': n_jobs,
                      'passes': n_passes
                      }
            lda_model = gensim.models.LdaMulticore(**params)
            lda_model.save(lda_file)
        return lda_model

    def build_lda_corpus(self, n_topics=100, passes=3, n_jobs=3):
        lda_file = os.path.join(self.tmp_folder, 'lda_corpus.mm')
        if os.pathisfile(lda_file):
            lda_corpus = gensim.corpora.mmcorpus.MmCorpus(lda_file)
        else:
            lda_model = self.build_lda_model(n_topics=100, passes=3, n_jobs=3)
            lda_corpus = lda_model[self.corpus]
            gensim.corpora.MmCorpus.serialize(lda_file, lda_corpus)
        return lda_corpus

    def build_hdp_model(self, max_topics):
        self.corpus
        self.dictionary
        pass

    def build_hdp_corpus(self, max_topics):
        self.corpus
        self.dictionary
        pass

if __name__ == '__main__':
    cwd = os.getcwd()
    tmp_folder = os.path.join(cwd, 'tmp', 'modeldir')
    semantic = TopicModels(tmp_folder)
