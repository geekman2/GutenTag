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

def graph_exponential():
    plt.plot([x**2 for x in range(50)])
    plt.xticks(range(0, 50, 5))
    plt.xlabel('Number of Features')
    plt.ylabel('Number of Samples Required')
    plt.show()

def get_distribution():
