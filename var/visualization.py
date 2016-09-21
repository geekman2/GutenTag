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

cwd = os.path.join(settings.project_root, 'tmp')

def graph_exponential():
    plt.plot([x**2 for x in range(50)])
    plt.xticks(range(0, 50, 5))
    plt.xlabel('Number of Features')
    plt.ylabel('Number of Samples Required')
    plt.show()

def get_distribution():
    distribution = {}
    counts = cPickle.load(open(os.path.join(cwd, 'counts.dict')))
    for label in counts:
        distribution[label] = {}
        trigrams = counts[label]
        total = sum(trigrams.values())
        for gram in trigrams:
            distribution[label][gram] = counts[label][gram]/float(total)
        sorted_distribution = sorted(distribution[label].items(), key=operator.itemgetter(0))
        distribution[label] = sorted_distribution
    for genre in counts:
        print(distribution[genre][:5])

get_distribution()