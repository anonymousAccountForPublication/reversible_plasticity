# waht we need: policy and state transition matrix
# combine those two into one ditcionary and read it out into a textfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import Counter
from PlotCircles import circles
import itertools
import os
import operator

def markovSim(mc, adultT, startDist):
    resultArr = []

    pE0E0, pE1E1 = mc
    pE0E1 = float(1- pE0E0)
    pE1E0 = float(1- pE1E1)

    P = np.array([[pE0E0,pE0E1],[pE1E0,pE1E1]])
    resultArr.append(round(startDist[1],3))
    for t in np.arange(0,adultT,1):
        newDist = np.dot(startDist,np.linalg.matrix_power(P,(t+1)))
        newDist = np.array(newDist)/float(sum(newDist))
        resultArr.append(round(newDist[1],3))
    return resultArr


def chunks(l, n):
    if n == 0:
        yield l
    else:
        # For item i in a range that is a length of l,
        for i in range(0, len(l), n):
            # Create an index range for l of n items:
            if isinstance(l, list):
                yield l[i:i+n]
            else:

                yield l.loc[i:i+n-1].reset_index(drop = True)

def convertValues(valueArr, old_max ,old_min,new_max, new_min):
    minArr =old_min
    maxArr = old_max
    rangeArr = maxArr-minArr
    newRangeArr = new_max-new_min
    result = [((val - minArr)/float(rangeArr))*newRangeArr+new_min for val in valueArr]
    return result



def area_calc(probs, r):
    result = [np.sqrt(float(p))*r for p in probs]
    return result

def duplicates(n):
    counter=Counter(n) #{'1': 3, '3': 3, '2': 3}
    dups=[i for i in counter if counter[i]!=1] #['1','3','2']
    result={}
    for item in dups:
        result[item]=[i for i,j in enumerate(n) if j==item]
    return result

# hepler function for plotting the lines
def isReachable(currentIdent, nextIdentList):

    condition_a = (currentIdent[0]+1, currentIdent[1])
    condition_b = (currentIdent[0], currentIdent[1]+1)

    yVals = [idx for idx,item in enumerate(nextIdentList) if (condition_a == item or condition_b == item)]
    yVals = list(set(yVals))
    return yVals


def joinIndidividualResultFiles(argument, tValues, dataPath):
    # need to provide the dataPath accordingly
    if argument == 'raw':

        resultsDFAll =[]
        for t in tValues:
            print('Currently aggregating data for time step %s' % t)

            # read and concatenate all csv file for one time step
            resultsDF = pd.read_csv(os.path.join(dataPath, os.path.join('%s.csv' % t)),index_col=0).reset_index(drop = True)
            resultsDFAll.append(resultsDF)

        finalData = pd.concat(resultsDFAll).reset_index(drop = True)
        finalData.to_csv('finalRaw.csv')

    elif argument == "aggregated":
        resultsDF = pd.concat(
            [pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % t), index_col=0) for t in
             tValues]).reset_index(drop=True)
        resultsDF.to_csv('finalAggregated.csv')
    elif argument == 'plotting':
        resultsDF = pd.concat(
            [pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' % t), index_col=0) for t in
             tValues]).reset_index(drop=True)
        resultsDF.to_csv('finalPlotting.csv')

    else:
        print("Wrong argument")

# function for parallelization
def plotLinesCopy(subDF1Identifiers, nextIdentList):
    yvalsIDXAll = []
    if subDF1Identifiers:
        for identSubDF1 in subDF1Identifiers:  # as many lines as unique cue validities
            subList = [isReachable(ident, nextIdentList) for ident in identSubDF1]
            subList.sort()
            subList2 = list(subList for subList, _ in itertools.groupby(subList))
            yvalsIDXAll.append(subList2)
            del subList
            del subList2
    return yvalsIDXAll

def plotLines(identSubDF1, nextIdentList):
    subList = [isReachable(ident, nextIdentList) for ident in identSubDF1]
    subList.sort()
    subList2 = list(subList for subList, _ in itertools.groupby(subList))
    del subList
    return subList2

def func_star(allArgs):
    return plotLines(*allArgs)


def cleanIdentifiers(oldIdentifiers):
    newIdent = [str(ident).replace('[', '').replace(']', '').split(',') for ident in oldIdentifiers]
    newIdent2 = [[int(str(a).replace('.0', '')) for a in subList] for subList in newIdent]

    return newIdent2

"""
write a function for an aggregated policy plot 

one plot per cue relaibility 
will first need to aggregate data across different cue reliablities 
cueValDict will contain three parameters now; then I can do the plotting

of course there is no aggregation when it comes to policies 

"""


def policyPlotReduced(T,r,priorE0Arr, pC0E0Arr, tValues, dataPath, lines, argumentR, argumentP, minProb,mainPath, plottingPath):
    # preparing the subplot
    fig, axes = plt.subplots(len(pC0E0Arr), len(priorE0Arr), sharex= True, sharey= True)
    fig.set_size_inches(16, 16)
    fig.set_facecolor("white")
    ax_list = fig.axes

    # looping over the paramter space
    iX = 0
    for cueVal in pC0E0Arr: # for each cue validity
        jX = 0
        for pE0 in priorE0Arr: # for each prior
            # set the working directory for the current parameter combination
            os.chdir(os.path.join(mainPath,"runTest_%s%s_%s%s" % (argumentR[0], argumentP[0], pE0,cueVal)))

            ax = ax_list[iX*len(priorE0Arr)+jX]
            plt.sca(ax)
            # preparing data for the pies
            coordinates = []
            decisionsPies = []
            stateProbPies = []

            for t in tValues:
                # here is where the relevant files are loaded
                aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' %t))


                # convert range to have a square canvas for plotting (required for the circle and a sensible aspect ratio of 1)
                aggregatedResultsDF['newpE1'] = convertValues(aggregatedResultsDF['pE1'], 1, 0, T - 1, 1)
                aggregatedResultsDF = aggregatedResultsDF[aggregatedResultsDF.stateProb >minProb] # minProb chance of reaching that state
                if t >= 1:
                    subDF = aggregatedResultsDF[aggregatedResultsDF['time'] ==t]
                    subDF = subDF.reset_index(drop=True)

                    pE1list = subDF['newpE1']
                    duplicateList = duplicates(pE1list)
                    if duplicateList:
                        stateProbs = list(subDF['stateProb'])
                        decisionMarker = list(subDF['marker'])
                        for key in duplicateList:
                            idxDuplList = duplicateList[key]
                            coordinates.append((t,key))
                            stateProbPies.append([stateProbs[i] for i in idxDuplList])
                            decisionsPies.append([decisionMarker[i] for i in idxDuplList])


                color_palette = {0:'#be0119', 1:'#448ee4', 2:'#8E44AD', 3: '#2b8214', 4: '#000000', -1: '#d8dcd6',
                                 5:'#7B3F00',6:'#dec912',7:'#FF8D79',8:'#79C4FF',9: '#8696a5', 10: '#566573',11:'#cc7722'}
                # red,blue, purple,dark green,black, chocolate brown, yellow, light red,light blue,mid gray, dark grey, ochre
                colors = np.array([color_palette[idx] for idx in aggregatedResultsDF['marker']])

                aggregatedResultsDF['area'] = area_calc(aggregatedResultsDF['stateProb'], r)
                area = aggregatedResultsDF['area']
                # now plot the developmental trajectories
                #sort this from largest to smallest ares

                circles(np.array(aggregatedResultsDF['time']),np.array(aggregatedResultsDF['newpE1']), s =np.array(aggregatedResultsDF['area']), ax = ax,c = colors, zorder = 2, lw = 0.5, alpha =.8)

                del aggregatedResultsDF
            # plotting the lines

            if lines:
                startTime = time.perf_counter()
                for t in np.arange(0,T-1,1):
                    print("Current time step: %s" % t)
                    tNext = t+1
                    timeArr = [t, tNext]

                    if t == 0:
                        plottingDF = pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' % (t+1)))
                        plottingDF['newpE1'] = convertValues(plottingDF['pE1'], 1, 0, T - 1, 1)

                        subDF1 = plottingDF[plottingDF['time'] == t]
                        subDF1 = subDF1.reset_index(drop=True)

                        subDF2 = plottingDF[plottingDF['time'] == tNext]
                        subDF2 = subDF2.reset_index(drop=True)

                        aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % (tNext)))
                        aggregatedResultsDF = aggregatedResultsDF[aggregatedResultsDF.time ==1]
                        aggregatedResultsDF = aggregatedResultsDF.reset_index(drop = True)
                        indices = aggregatedResultsDF.index[aggregatedResultsDF.stateProb > minProb].tolist()
                        subDF2 = subDF2.iloc[indices]
                        subDF2 = subDF2.reset_index(drop=True)
                        del aggregatedResultsDF

                    else:

                        subDF1 = subDF2
                        del subDF2

                        aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % (tNext)))
                        aggregatedResultsDF.drop_duplicates(subset='pE1', inplace=True)
                        aggregatedResultsDF.reset_index(drop=True, inplace=True)
                        indices = aggregatedResultsDF.index[aggregatedResultsDF.stateProb <= minProb].tolist()
                        del aggregatedResultsDF

                        subDF2 = pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' %tNext))
                        subDF2['newpE1'] = convertValues(subDF2['pE1'], 1, 0, T - 1, 1)
                        subDF2.reset_index(drop=True, inplace= True)

                        subDF2.drop(index = indices, inplace= True)
                        subDF2.reset_index(drop=True, inplace=True)
                        del indices

                    nextIdent = tuple(zip(list(subDF2.x0), list(subDF2.x1)))
                    currIdent = tuple(zip(list(subDF1.x0), list(subDF1.x1)))


                    yvalsIDXAll = [isReachable(identSubDF1, nextIdent) for identSubDF1 in currIdent]

                    # process the results
                    for subIDX in range(len(subDF1)):
                        yArr = [[subDF1['newpE1'].loc[subIDX], subDF2['newpE1'].loc[yIDX]] for yIDX in
                                yvalsIDXAll[subIDX]]
                        [ax.plot(timeArr, yArrr, ls='solid', marker=" ", color='#e6daa6', zorder=1, lw=0.3) for yArrr in
                         yArr]
                        del yArr
                elapsedTime = time.perf_counter()-startTime
                print("Elapsed time plotting the lines: " + str(elapsedTime))

            # next step adding pies for cases where organisms with the same estimates make different decisions
            # this does not check whether the decisions are actually different; it does so implicitly

            for idx in range(len(coordinates)):
                colorsPies = [color_palette[idj] for idj in decisionsPies[idx]]
                pieFracs = [float(i) / sum(stateProbPies[idx]) for i in stateProbPies[idx]]
                currentR = np.sqrt(sum(stateProbPies[idx])) * r
                pp, tt = ax.pie(pieFracs, colors=colorsPies, radius=currentR, center=coordinates[idx],
                                wedgeprops={'linewidth': 0.0, "edgecolor": "k"})
                [p.set_zorder(3 + len(coordinates) - idx) for p in pp]
                ax.set_frame_on(True)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()


            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)



            plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
            midPoint = (T) / float(2)
            yLabels = convertValues([1, midPoint, T - 1], T - 1, 1, 1, 0)

            # removing frame around the plot
            plt.ylim(0.4, T - 1 + 0.5)
            plt.xlim(-0.6, T - 1 + 0.5)

            if iX == 0:
                plt.title(1-pE0, fontsize=20)

            if jX == 0:
                plt.yticks([1, midPoint, T - 1], yLabels, fontsize=15)
                plt.ylabel('estimate', fontsize=20, labelpad=10)

            if jX == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), fontsize=20, labelpad=15, rotation='vertical')
                ax.yaxis.set_label_position("right")

            if iX == (len(pC0E0Arr)-1):
                plt.xticks(np.arange(0, max(tValues) + 1, 2), fontsize=15)
                plt.xlabel('ontogeny', fontsize =20, labelpad=15)

            if iX < (len(pC0E0Arr) - 1):
                plt.tick_params(bottom=False)

            if jX > 0:
                plt.tick_params(left=False)

            ax.set_aspect('equal')

            jX += 1
        iX += 1
    plt.suptitle('prior probability', fontsize = 20)
    fig.text(0.98,0.5,'cue reliability', fontsize = 20, horizontalalignment = 'right', verticalalignment = 'center', rotation = 'vertical')
    resultPath = os.path.join(mainPath, plottingPath)
    #plt.savefig(os.path.join(resultPath,'DevelopmentalTrajectory.pdf'), dpi = 900)
    plt.savefig(os.path.join(resultPath, 'DevelopmentalTrajectoryReduced.png'), dpi=600)




def policyPlotReduced_v2(T,r,priorE0Arr, pC0E0Arr, tValues, dataPath, lines, argumentR, argumentP, minProb,mainPath, plottingPath):
    # preparing the subplot
    fig, axes = plt.subplots(len(pC0E0Arr), len(priorE0Arr), sharex= True, sharey= True)
    fig.set_size_inches(16, 16)
    fig.set_facecolor("white")
    ax_list = fig.axes

    # looping over the paramter space
    iX = 0
    for cueVal in pC0E0Arr: # for each cue validity
        jX = 0
        for pE0 in priorE0Arr: # for each prior
            # set the working directory for the current parameter combination
            os.chdir(os.path.join(mainPath,"runTest_%s%s_%s%s" % (argumentR[0], argumentP[0], pE0,cueVal)))

            ax = ax_list[iX*len(priorE0Arr)+jX]
            plt.sca(ax)
            # preparing data for the pies
            coordinates = []
            decisionsPies = []
            stateProbPies = []

            for t in tValues:
                # here is where the relevant files are loaded
                aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' %t))


                # convert range to have a square canvas for plotting (required for the circle and a sensible aspect ratio of 1)
                aggregatedResultsDF['newpE1'] = convertValues(aggregatedResultsDF['pE1'], 1, 0, T - 1, 1)
                aggregatedResultsDF = aggregatedResultsDF[aggregatedResultsDF.stateProb >minProb] # minProb chance of reaching that state
                if t >= 1:
                    subDF = aggregatedResultsDF[aggregatedResultsDF['time'] ==t]
                    subDF = subDF.reset_index(drop=True)

                    pE1list = subDF['newpE1']
                    duplicateList = duplicates(pE1list)
                    if duplicateList:
                        stateProbs = list(subDF['stateProb'])
                        decisionMarker = list(subDF['marker'])
                        for key in duplicateList:
                            idxDuplList = duplicateList[key]
                            coordinates.append((t,key))
                            stateProbPies.append([stateProbs[i] for i in idxDuplList])
                            decisionsPies.append([decisionMarker[i] for i in idxDuplList])


                color_palette = {0:'#be0119', 1:'#448ee4', 2:'#79C4FF', 3: '#FF8D79', 4: '#000000', -1: '#d8dcd6',
                                 5:'#8E44AD',6:'#d39beb',7:'#d39beb',8:'#d39beb',9: '#d39beb', 10: '#d39beb',11:'#cc7722'}
                # red,blue, purple,dark green,black, chocolate brown, yellow, light red,light blue,mid gray, dark grey, ochre
                colors = np.array([color_palette[idx] for idx in aggregatedResultsDF['marker']])

                aggregatedResultsDF['area'] = area_calc(aggregatedResultsDF['stateProb'], r)
                area = aggregatedResultsDF['area']
                # now plot the developmental trajectories
                #sort this from largest to smallest ares

                circles(np.array(aggregatedResultsDF['time']),np.array(aggregatedResultsDF['newpE1']), s =np.array(aggregatedResultsDF['area']), ax = ax,c = colors, zorder = 2, lw = 0.5, alpha =.8)

                del aggregatedResultsDF
            # plotting the lines

            if lines:
                startTime = time.perf_counter()
                for t in np.arange(0,T-1,1):
                    print("Current time step: %s" % t)
                    tNext = t+1
                    timeArr = [t, tNext]

                    if t == 0:
                        plottingDF = pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' % (t+1)))
                        plottingDF['newpE1'] = convertValues(plottingDF['pE1'], 1, 0, T - 1, 1)

                        subDF1 = plottingDF[plottingDF['time'] == t]
                        subDF1 = subDF1.reset_index(drop=True)

                        subDF2 = plottingDF[plottingDF['time'] == tNext]
                        subDF2 = subDF2.reset_index(drop=True)

                        aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % (tNext)))
                        aggregatedResultsDF = aggregatedResultsDF[aggregatedResultsDF.time ==1]
                        aggregatedResultsDF = aggregatedResultsDF.reset_index(drop = True)
                        indices = aggregatedResultsDF.index[aggregatedResultsDF.stateProb > minProb].tolist()
                        subDF2 = subDF2.iloc[indices]
                        subDF2 = subDF2.reset_index(drop=True)
                        del aggregatedResultsDF

                    else:

                        subDF1 = subDF2
                        del subDF2

                        aggregatedResultsDF = pd.read_csv(os.path.join(dataPath, 'aggregatedResults_%s.csv' % (tNext)))
                        aggregatedResultsDF.drop_duplicates(subset='pE1', inplace=True)
                        aggregatedResultsDF.reset_index(drop=True, inplace=True)
                        indices = aggregatedResultsDF.index[aggregatedResultsDF.stateProb <= minProb].tolist()
                        del aggregatedResultsDF

                        subDF2 = pd.read_csv(os.path.join(dataPath, 'plottingResults_%s.csv' %tNext))
                        subDF2['newpE1'] = convertValues(subDF2['pE1'], 1, 0, T - 1, 1)
                        subDF2.reset_index(drop=True, inplace= True)

                        subDF2.drop(index = indices, inplace= True)
                        subDF2.reset_index(drop=True, inplace=True)
                        del indices

                    nextIdent = tuple(zip(list(subDF2.x0), list(subDF2.x1)))
                    currIdent = tuple(zip(list(subDF1.x0), list(subDF1.x1)))


                    yvalsIDXAll = [isReachable(identSubDF1, nextIdent) for identSubDF1 in currIdent]

                    # process the results
                    for subIDX in range(len(subDF1)):
                        yArr = [[subDF1['newpE1'].loc[subIDX], subDF2['newpE1'].loc[yIDX]] for yIDX in
                                yvalsIDXAll[subIDX]]
                        [ax.plot(timeArr, yArrr, ls='solid', marker=" ", color='#e6daa6', zorder=1, lw=0.3) for yArrr in
                         yArr]
                        del yArr
                elapsedTime = time.perf_counter()-startTime
                print("Elapsed time plotting the lines: " + str(elapsedTime))

            # next step adding pies for cases where organisms with the same estimates make different decisions
            # this does not check whether the decisions are actually different; it does so implicitly

            for idx in range(len(coordinates)):
                colorsPies = [color_palette[idj] for idj in decisionsPies[idx]]
                pieFracs = [float(i) / sum(stateProbPies[idx]) for i in stateProbPies[idx]]
                currentR = np.sqrt(sum(stateProbPies[idx])) * r
                pp, tt = ax.pie(pieFracs, colors=colorsPies, radius=currentR, center=coordinates[idx],
                                wedgeprops={'linewidth': 0.0, "edgecolor": "k"})
                [p.set_zorder(3 + len(coordinates) - idx) for p in pp]
                ax.set_frame_on(True)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()


            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)



            plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
            midPoint = (T) / float(2)
            yLabels = convertValues([1, midPoint, T - 1], T - 1, 1, 1, 0)

            # removing frame around the plot
            plt.ylim(0.4, T - 1 + 0.5)
            plt.xlim(-0.6, T - 1 + 0.5)

            if iX == 0:
                plt.title(1-pE0, fontsize=20)

            if jX == 0:
                plt.yticks([1, midPoint, T - 1], yLabels, fontsize=15)
                plt.ylabel('estimate', fontsize=20, labelpad=10)

            if jX == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), fontsize=20, labelpad=15, rotation='vertical')
                ax.yaxis.set_label_position("right")

            if iX == (len(pC0E0Arr)-1):
                plt.xticks(np.arange(0, max(tValues) + 1, 2), fontsize=15)
                plt.xlabel('ontogeny', fontsize =20, labelpad=15)

            if iX < (len(pC0E0Arr) - 1):
                plt.tick_params(bottom=False)

            if jX > 0:
                plt.tick_params(left=False)

            ax.set_aspect('equal')

            jX += 1
        iX += 1
    plt.suptitle('prior probability', fontsize = 20)
    fig.text(0.98,0.5,'cue reliability', fontsize = 20, horizontalalignment = 'right', verticalalignment = 'center', rotation = 'vertical')
    resultPath = os.path.join(mainPath, plottingPath)
    #plt.savefig(os.path.join(resultPath,'DevelopmentalTrajectory.pdf'), dpi = 900)
    plt.savefig(os.path.join(resultPath, 'DevelopmentalTrajectoryReduced_simplified.png'), dpi=600)



