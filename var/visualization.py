#-------------------------------------------------------------------------------
# Name:         Visualization
# Purpose:      Generate visualizations for report
# Author:       Devon Muraoka
# Created:      9/20/2016
# Copyright:   (c) Devon Muraoka, Bharat Ramanathan 
#-------------------------------------------------------------------------------
from __future__ import absolute_import, print_function
import seaborn
from matplotlib import pyplot as plt
import cPickle
import settings
import os
import operator
from pprint import pprint
import numpy as np

cwd = os.path.join(settings.project_root, 'tmp')

def graph_exponential():
    plt.plot([x**2 for x in range(50)])
    plt.xticks(range(0, 50, 5))
    plt.xlabel('Number of Features')
    plt.ylabel('Number of Samples Required')
    plt.show()

def plot_data(frequency, label):
    X = np.arange(len(frequency))
    plt.bar(X, frequency, align='center', width=0.5)
    plt.xticks(X, label)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    ymax = max(frequency) + 0.001
    plt.ylim(0, ymax)
    plt.show()

def get_distribution():
    fig = plt.figure()
    distribution = {}
    counts = cPickle.load(open(os.path.join(cwd, 'counts.dict')))
    for label in counts:
        distribution[label] = {}
        trigrams = counts[label]
        total = sum(trigrams.values())
        for gram in trigrams:
            new_value = counts[label][gram]/float(total)
            distribution[label][gram] = new_value
        sorted_distribution = sorted(distribution[label].items(), key=operator.itemgetter(1), reverse=True)
        distribution[label] = sorted_distribution
    for i, genre in enumerate(counts):
        genre_dist = distribution[genre][:10]
        frequency = [x[1] for x in genre_dist]
        labels = [y[0] for y in genre_dist]
        print(genre+':')
        pprint(genre_dist)

        X = np.arange(len(frequency))
        plt.bar(X, frequency, align='center', width=0.5)
        plt.xticks(X, labels)
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=0)
        ymax = max(frequency) + 0.001
        plt.ylim(0, ymax)
        plt.title(genre)
        plt.tight_layout()
        plt.show()



get_distribution()