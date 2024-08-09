import numpy as np
import os
import pandas as pd
import time
from multiprocessing import Pool
import itertools
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.markers import MarkerStyle
import pickle
import ternary
import seaborn as sns
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sympy.utilities.iterables import multiset_permutations
from decimal import Decimal as D
import operator
from scipy.stats import binom
import scipy.special

def plotFitnessDifference(priorE0Arr, cueValidityArr, resultsNoD, resultsIncrD, resultsAbruptD):
    patterns = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]

    # first open the dictionary containing the results
    differencesDictNoD = pickle.load(open(os.path.join(resultsNoD, "fitnessDifferences.p"), "rb"))
    differencesDictIncrD = pickle.load(open(os.path.join(resultsIncrD, "fitnessDifferences.p"), "rb"))
    differencesDictAbruptD = pickle.load(open(os.path.join(resultsAbruptD, "fitnessDifferences.p"), "rb"))

    # define the xAxis
    x = np.arange(5)
    xLabels = ["S", "O\nno D","O\nincr. D","O\ncompl. D", "G"]
    fig, axes = plt.subplots(len(cueValidityArr), len(priorE0Arr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]
            ax.set(aspect="equal")
            plt.sca(ax)
            # open the relevant fitness difference array
            fitnessDifferencesnoD = differencesDictNoD[(pE0, cueVal)]
            fitnessDifferencesIncrD = differencesDictIncrD[(pE0, cueVal)]
            fitnessDifferencesAbruptD = differencesDictAbruptD[(pE0, cueVal)]

            fitnessDifferences = [fitnessDifferencesAbruptD[0],fitnessDifferencesnoD[1],fitnessDifferencesIncrD[1],fitnessDifferencesAbruptD[1],fitnessDifferencesAbruptD[2]]

            barList = plt.bar(x, fitnessDifferences)

            barList[0].set_color("lightgray")
            #barList[0].set_edgecolor("black")
            barList[1].set_color("grey")
            barList[1].set_hatch(patterns[0])
            barList[1].set_edgecolor("black")
            barList[2].set_color("grey")
            barList[2].set_hatch(patterns[5])
            barList[2].set_edgecolor("black")
            barList[3].set_color("grey")
            barList[3].set_hatch(patterns[8])
            barList[3].set_edgecolor("black")
            barList[4].set_color("black")
            #barList[4].set_edgecolor("black")

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(False)

            plt.ylim(-1, 1)

            if ix == 0:
                plt.title("%s" % (1 - pE0), fontsize=20)

            if ix == len(cueValidityArr) - 1:
                plt.xlabel('', fontsize=20, labelpad=10)
                plt.xticks(x, xLabels, fontsize=15, rotation=45)

            else:
                ax.get_xaxis().set_visible(False)
            if jx == 0:
                plt.ylabel('Fitness difference', fontsize=20, labelpad=10)
                plt.yticks([-1, 0, 1], fontsize=15)

            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=15, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")
            ax.set(aspect=2)
            jx += 1
        ix += 1

        plt.suptitle('prior', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 rotation='vertical')
    #plt.savefig(os.path.join(resultsAbruptD, 'fitnessDifferencesMerged.pdf'), dpi=600)
    plt.savefig(os.path.join(resultsAbruptD, 'fitnessDifferencesMerged.png'), dpi=600)

    plt.close()


# resultsNoD = "/home/nicole/Projects/2016model/5_ts/"
# resultsIncrD = "/media/nicole/Elements/Results_reversible_development_model/5_ts/"
# resultsAbruptD = "/media/nicole/Elements/abrupt_small_TS/5_ts/"
# mainPath = "/media/nicole/Elements/abrupt_small_TS/5_ts/"


resultsNoD = "/home/nicole/Projects/2016model/20_timesteps/"
resultsIncrD = "/media/nicole/Elements/Results_reversible_development_model/newResults/"
resultsAbruptD = "/media/nicole/Elements/ReversibleDevelopmentAbrupt/"
mainPath = "/media/nicole/Elements/ReversibleDevelopmentAbrupt/"


priorE0Arr = [0.5, 0.3, 0.1]  #
# corresponds to the probability of receiving C0 when in E0
cueValidityC0E0Arr = [0.55, 0.75, 0.95]  #

argumentRArr = ['linear','diminishing', 'increasing']
argumentPArr = ['linear', 'diminishing', 'increasing']

for argumentR in argumentRArr:
    for argumentP in argumentPArr:
        print("Plot for reward " + str(argumentR) + " and penalty " + str(argumentP))
        twinResultsPathNoD = os.path.join(resultsNoD,"PlottingResults_%s_%s" % (argumentR[0], argumentP[0]))
        twinResultsPathIncrD = os.path.join(resultsIncrD, "PlottingResults_%s_%s" % (argumentR[0], argumentP[0]))
        twinResultsPathAbruptD = os.path.join(resultsAbruptD, "PlottingResults_%s_%s" % (argumentR[0], argumentP[0]))
        plotFitnessDifference(priorE0Arr, cueValidityC0E0Arr, twinResultsPathNoD, twinResultsPathIncrD, twinResultsPathAbruptD)