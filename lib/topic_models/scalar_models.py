import os
from sklearn import manifold
from time import time


class DimeReducer(object):
    def __init__(self, data):
        self.data = data

    def get_mds_scalar(self, data):
        mds = manifold.MDS(n_components=2,
                           dissimilarity='precomputed',
                           n_jobs=-1,
                           random_state=1
                           )
        data = mds.fit_transform(data)
        return data


if __name__ == '__main__':
    from lib.topic_models.vector_models import VectorModels
    from lib.topic_models.semantic_models import TopicModels
    from lib.topic_models.distance_metrics import DistanceMetrics

    start_corpus = time()
    cwd = os.getcwd()
    data_loc = os.path.join(cwd, 'tmp', 'text_corpus.dat', )
    tmp_folder = os.path.join(cwd, 'tmp', 'modeldir')

    vectors = VectorModels(data_loc, tmp_folder)
    corpus, dictionary = vectors.load_corpus()
    tfidf_corpus = vectors.build_tfidf_corpus(corpus, dictionary)

    semantic = TopicModels(tmp_folder, tfidf_corpus, dictionary)
    lda_corpus = semantic.build_lda_corpus(bow=False)

    metrics = DistanceMetrics(tmp_dir=tmp_folder, corpus=lda_corpus)
    sims = metrics.build_cosine_similarity_index()


    reducer = DimeReducer(sims)
    reduced_data = reducer.get_mds_scalar(sims)

    print(reduced_data)
    stop_corpus = time()
    corpus_time = round(stop_corpus - start_corpus, 3)
    logger.info('time taken to build corpus = {}s'.format(corpus_time))
