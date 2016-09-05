import pandas as pd
import logging
import os
from matplotlib import pylab as plt
import seaborn as sns
sns.set(color_codes=True)

cwd = os.getcwd()
working_directory = os.path.join(cwd,'tmp','modeldir')
if not os.path.exists(working_directory):
    os.makedirs(working_directory)

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def plot_clusters(plot_data, cluster_data):
    xs, ys = plot_data[:, 0], plot_data[:, 1]
    df = pd.DataFrame({"clusters": cluster_data, "Xs": xs, "ys": ys})
    docgroups = df.groupby("clusters")
    logger.info('Plotting the Clusters')
    plt.figure(figsize=(20, 10))
    for name, group in docgroups:
        plt.plot(group.Xs, group.ys, marker='o', ms=10, linestyle='', label=name)
    plt.show()


if __name__ == '__main__':
    from lib.pre_processing import CorpusModel, SemanticModels, SimilarityModel, Clusterer, ReduceDimension

    corpus_model = CorpusModel()
    dictionary = corpus_model.load_dict()
    corpus = corpus_model.load_corpus()
    tfidf_corpus = corpus_model.load_tfidf_corpus()
    semantic_model = SemanticModels(corpus=tfidf_corpus, dictionary=dictionary, tfidf=True)
    tfidf_lda = semantic_model.load_lda_model()
    sims_model = SimilarityModel(corpus=tfidf_corpus, dictionary=dictionary, tfidf=True, model=None)
    sim_index = sims_model.load_sim_index()
    cluster_model = Clusterer(doc_vecs=sim_index, tfidf=True, model=None)
    cluster_data = cluster_model.k_clusterer()
    reducer = ReduceDimension(data=sim_index, tfidf=True, model=None)
    plot_data = reducer.t_sne_reduce()
    plot_clusters(plot_data, cluster_data)
