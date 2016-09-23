from sklearn.metrics import adjusted_rand_score, v_measure_score
import settings
import os
import numpy as np

from time import time


def get_labels():
    labels_file = os.path.join(settings.project_root, 'tmp', 'text_labels.dat')
    labels = [line.strip() for line in open(labels_file)]
    map_dict = {key: value for value, key in enumerate(set(labels))}
    labels = np.vectorize(map_dict.get)(np.array(labels))
    return labels

if __name__ == '__main__':
    from lib.topic_models.vector_models import VectorModels
    from lib.topic_models.semantic_models import TopicModels
    from lib.topic_models.cluster_models import Clusterer
    start_corpus = time()
    data_loc = os.path.join(settings.project_root, 'tmp', 'text_corpus.dat', )
    tmp_folder = os.path.join(settings.project_root, 'tmp', 'modeldir')

    vectors = VectorModels(data_loc, tmp_folder)
    corpus, dictionary = vectors.load_corpus()
    tfidf_corpus = vectors.build_tfidf_corpus(corpus, dictionary)

    semantic = TopicModels(tmp_folder, tfidf_corpus, dictionary)
    lda_corpus = semantic.build_lda_corpus(bow=False)

    clustering = Clusterer(tmp_folder,lda_corpus, n_docs=lda_corpus.num_docs)
    k_labels, k_means = clustering.mini_k_clusters(k=5, corpus_type='lda_tfidf')
    print("Adjusted Rand Score = {}".format(adjusted_rand_score(get_labels(), k_labels)))
    print("V Measure Score = {}".format(v_measure_score(get_labels(), k_labels)))