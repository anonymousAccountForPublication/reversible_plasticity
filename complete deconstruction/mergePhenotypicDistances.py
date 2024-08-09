import numpy as np
import os
import matplotlib.pyplot as plt
import pickle




def plotDistances(tValues, priorE0Arr, cueValidityArr, absoluteDistanceDictIncrD,
                  absoluteDistanceDictAbruptD,
                  argument, adoptionType, lag, endOfExposure, VarArg,
                  env,twinResultsPath):
    arg = "absolute"  # choose whether you want this plot for relative or absolute distance

    linestyle_tuple = [
        ('solid', (0, ())),
        ('dotted', (0, (1, 1))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),
        ('densely dashed', (0, (5, 1))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),
        ('loosely dotted', (0, (1, 10))),
        ('loosely dashed', (0, (5, 10))),
        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('densely dotted', (0, (1, 1))),
        ('dashed', (0, (5, 5))),
        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)))]

    markerStyles = ['o','*','s']
    fig, axes = plt.subplots(len(cueValidityArr), len(priorE0Arr), sharex=True, sharey=True)
    fig.set_size_inches(16.5, 20)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]
            ax.set(aspect=max(tValues) - 1)
            plt.sca(ax)
            absoluteDistanceI = absoluteDistanceDictIncrD[(pE0, cueVal)]

            absoluteDistanceA = absoluteDistanceDictAbruptD[(pE0, cueVal)]


            [plt.plot(tValues, keyVal[1], linestyle=linestyle_tuple[idx][1], linewidth=2.5, markersize=6,
                      marker=markerStyles[idx], color='#298c8c', markerfacecolor='#298c8c',
                      label="%s incremental" % (keyVal[0])) for idx, keyVal in enumerate(absoluteDistanceI.items())]


            [plt.plot(tValues, keyVal[1], linestyle=linestyle_tuple[idx][1], linewidth=2.5, markersize=6,
                      marker=markerStyles[idx], color='#f1a226', markerfacecolor='#f1a226',
                      label="%s complete" % (keyVal[0])) for idx, keyVal in enumerate(absoluteDistanceA.items())]

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)

            plt.ylim(-0.05, 1.1)
            plt.yticks(np.arange(0, 1.1, 0.2), fontsize=15)
            plt.xticks(np.arange(1, max(tValues) + 1, 2), fontsize=15)

            if ix == len(cueValidityArr) - 1 and jx == 0:

                # reordering the legend labels
                handles, labels = plt.gca().get_legend_handles_labels()
                order = [0, 3,1,4,2,5]

                legend = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                                   loc='upper center', bbox_to_anchor=(1.75, -0.2),
                                   title='phenotypic dimensions',
                                   ncol=3, fancybox=True, shadow=False, fontsize=20)
                plt.setp(legend.get_title(), fontsize='20')
            if ix == 0:
                plt.title("%s" % (1 - pE0), fontsize=20)

            if (ix == (len(cueValidityArr) - 1)) and (jx == 0):
                plt.xlabel("ontogeny", fontsize=20, labelpad=10)
                plt.ylabel('phenotypic distance', fontsize=20, labelpad=10)

            if ix < (len(cueValidityArr) - 1):
                plt.tick_params(bottom=False)

            if jx > 0:
                plt.tick_params(left=False)

            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            jx += 1
        ix += 1
        plt.suptitle('prior estimate', fontsize=20, y=0.95)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center'
                 , rotation='vertical')
        if endOfExposure:
            safeStr = "EndOfExposure"
        else:
            safeStr = "EndOfOntogeny"

    # plt.savefig(
    #     os.path.join(twinResultsPath,
    #                  '%s_%s_%s_%s_%sPlasticity%s_%s_merged.pdf' % (argument, adoptionType, lag, safeStr, env, VarArg,arg)),
    #     dpi=600)
    plt.savefig(
        os.path.join(twinResultsPath,
                     '%s_%s_%s_%s_%sPlasticity%s_%s_merged.png' % (argument, adoptionType, lag, safeStr, env, VarArg,arg)),
        dpi=600)
    plt.close()



resultsIncrD = "/home/nicole/Projects/Results_reversible_development_model/newResults/"
resultsAbruptD = "/media/nicole/Elements/ReversibleDevelopmentAbrupt/"
mainPath = "/media/nicole/Elements/ReversibleDevelopmentAbrupt/"


# resultsIncrD = "/media/nicole/Elements/Results_reversible_development_model/10_ts/"
# resultsAbruptD = "/media/nicole/Elements/abrupt_small_TS/10_ts/"
# mainPath = "/media/nicole/Elements/abrupt_small_TS/10_ts/"

priorE0Arr = [0.5, 0.3, 0.1]  #
# corresponds to the probability of receiving C0 when in E0
cueValidityC0E0Arr = [0.55, 0.75, 0.95]  #

argumentRArr = ['linear','diminishing', 'increasing']
argumentPArr = ['linear', 'diminishing', 'increasing']
T =20

tValues = np.arange(1, T + 1, 1)

plotType = 'ExperimentalTwinstudy' #Twinstudy, ExperimentalTwinstudy
adoptionType = 'yoked' #yoked ,oppositePatch, deprivation
lag = 3 #None, 3
endOfExposure = True

for argumentR in argumentRArr:
    for argumentP in argumentPArr:
        print("Plot for reward " + str(argumentR) + " and penalty " + str(argumentP))
        twinResultsPathIncrD = os.path.join(resultsIncrD, "PlottingResults_%s_%s" % (argumentR[0], argumentP[0]))
        twinResultsPathAbruptD = os.path.join(resultsAbruptD, "PlottingResults_%s_%s" % (argumentR[0], argumentP[0]))



        absoluteDistanceDictIncrD = pickle.load(open(os.path.join(twinResultsPathIncrD, 'absoluteDistanceDict%s%s%s%s_%s.p' % (
            plotType, adoptionType, lag, endOfExposure, 1)), 'rb'))

        absoluteDistanceDictAbruptD = pickle.load(open(os.path.join(twinResultsPathAbruptD , 'absoluteDistanceDict%s%s%s%s_%s.p' % (
            plotType, adoptionType, lag, endOfExposure,1)), 'rb'))

        if lag:
            T_inside = max(tValues) - lag + 1
        else:
            T_inside = max(tValues)

        plotDistances(np.arange(1, T_inside + 1, 1), priorE0Arr, cueValidityC0E0Arr, absoluteDistanceDictIncrD,
                      absoluteDistanceDictAbruptD, plotType, adoptionType, lag, endOfExposure,False, 1,twinResultsPathAbruptD)

