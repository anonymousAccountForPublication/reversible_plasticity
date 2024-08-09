# replication of Willem's 2016 paper "The evolution of sensitive periods in a model of incremental development"
import sys
import numpy as np
import itertools
from numpy import random
import pickle
import time
import os
import matplotlib.pyplot as plt
from scipy.stats import binom
from decimal import Decimal as D
from PreparePlottingVarCueValidityParallel import preparePlotting
from PlottingParallel import joinIndidividualResultFiles, policyPlotReduced, policyPlotReduced_v2
from itertools import combinations
from twinStudiesVarCue3NewFastSim import runPlots


# Parameter description
# x0 : number of cues indicating Environment 0 at time t
# x1 : number of cues indicating Environment 1 at time t
# y0 : number of developmental steps at time t towards phenotype 0
# y1 : number of developmental steps at time t towards phenotype 1
# yw : number of developmental steps at time t in which the organism waited


# function to set global variables
def set_global_variables(aVar, bVar, probVar, funVar, probMax, probMin):
    # beta defines the curvature of the reward and penalty functions
    global aFT
    global bFT
    global probFT
    global funFT
    global probMaxFT
    global probMinFT
    global Ft1

    aFT = aVar
    bFT = bVar
    probFT = probVar
    funFT = funVar
    probMaxFT = probMax
    probMinFT = probMin


def set_global_Ft1(setFt1):
    global Ft1
    Ft1 = setFt1


# helper functions
def up():
    # My terminal breaks if we don't flush after the escape-code
    sys.stdout.write('\x1b[1A')
    sys.stdout.flush()


def down():
    # I could use '\x1b[1B' here, but newline is faster and easier
    sys.stdout.write('\n')
    sys.stdout.flush()


def isclose(pair, rel_tol=1e-09, abs_tol=0.0):
    a,b = pair
    rel_tol = float(rel_tol)
    abs_tol = float(abs_tol)
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def findOptimumClose(fitnesList):
    # fitnesList corresponds to [F0,F1,Fw]
    # determine whether list contains ties
    F0C = float(fitnesList[0])
    F1C = float(fitnesList[1])
    F0D = float(fitnesList[2])
    F1D = float(fitnesList[3])
    Fw = float(fitnesList[4])

    relTol = 1e-12

    maxVal = np.max(fitnesList) # this is the max
    closeIDX = [idx for idx, elem in enumerate(fitnesList) if isclose((maxVal,elem), rel_tol=relTol)]
    idx = random.choice(closeIDX)
    if len(closeIDX) > 1:
        secondArg =tuple([int(curr) for curr in closeIDX])
    else:
        secondArg = int(closeIDX[0])
    return fitnesList[idx], secondArg


def findOptimum(fitnesList):
    # fitnesList corresponds to [F0,F1,Fw]
    # determine whether list contains ties
    F0 = fitnesList[0]
    F1 = fitnesList[1]
    Fw = fitnesList[2]

    if F0 == F1 == Fw:
        idx = random.choice([0, 1, 2])
        return fitnesList[idx], (int(0), int(1), int(2))

    elif (F0 == F1 and F0 > Fw):
        idx = random.choice([0, 1])
        return fitnesList[idx], (int(0), int(1))

    elif (F0 == Fw and F0 > F1):
        idx = random.choice([0, 2])
        return fitnesList[idx], (int(0), int(2))

    elif (F1 == Fw and F1 > F0):
        idx = random.choice([1, 2])
        return fitnesList[idx], (int(1), int(2))
    else:
        maxVal = np.max(fitnesList)
        idx = np.argmax(fitnesList)
        return maxVal, int(idx)


def normData(data, MinFT, MaxFT):
    if min(data) < MinFT or max(data) > MaxFT:
        rangeArr = max(data) - min(data)
        newRangeArr = MaxFT - MinFT
        normalized = [((val - min(data)) / float(rangeArr)) * newRangeArr + MinFT for val in data]
        return normalized
    else:
        return data


def stepFun(t, a, b, prob, T):
    if t < a:
        return 0
    elif t <= b:
        merke = sum(np.arange(a, b + 1, 1))

        return t / (float(merke) / (1 - prob))
    else:
        return prob / float(T - b)


def probFinalTime(t, T, kw):
    # currently these are all summing to 1, but this is not a necessity?
    if kw == 'None':
        return 0
    elif kw == 'uniform':
        return 1 / float(T)
    elif kw == 'log':
        tVal = np.arange(1, T + 1, 1)
        return np.log(t) / float(sum(np.log(tVal)))
    elif kw == 'step':
        return stepFun(t, aFT, bFT, probFT, T)

    elif kw == 'fun':
        return funFT(t)
    else:
        print('Unkown keyword for random final time')
        exit(1)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def plotCueReliability(tValues, cueReliabilityArr, kw):
    plt.figure()
    ax = plt.gca()
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(tValues, cueReliabilityArr)
    plt.xlim(1, max(tValues))
    plt.title('Cue reliability as a function of time')
    plt.xlabel('Time step')
    plt.ylabel('Cue reliability')
    plt.savefig("CueReliability%s.png" % kw)
    plt.close("all")


def processCueValidity(pC0E0, tValues):
    if is_number(pC0E0):
        cueValidityE0Dict = {t: round(float(pC0E0), 3) for t in tValues}
        cueValidityE1Dict = cueValidityE0Dict
        return (cueValidityE0Dict, cueValidityE1Dict)
    else:
        print("cue reliability is not a number")
        exit(1)


# Bayesian updating scheme
def BayesianUpdating(pE0, pE1, pDE0, pDE1):
    pE0, pE1, pDE0, pDE1 = float(pE0), float(pE1), float(pDE0), float(pDE1)
    # pE0 is the evolutionary prior vor environment 1
    # pE1 is the evolutionary prior for environment 2
    # pDE0 and pDE1 are the probabilities of obtaining the data given environment 0 or 1 respectively (likelihood)
    p_D = pDE0 * pE0 + pDE1 * pE1
    b0_D = (pDE0 * pE0) / p_D
    b1_D = (pDE1 * pE1) / p_D

    return b0_D, b1_D


def fitnessFunc(state, b0_D, b1_D, argumentR, argumentP, T, beta, psi_weighting):
    x0, x1, y0, y1, yw = state

    if argumentR == 'linear':
        phiVar = b0_D * y0 + b1_D * y1

    elif argumentR == 'diminishing':
        alphaRD = (T) / float(1 - float(np.exp(-beta * (T))))
        alphaRD = alphaRD
        phiVar = b0_D * alphaRD * (1 - np.exp(-beta * y0)) + b1_D * alphaRD * (1 - np.exp(-beta * y1))

    elif argumentR == 'increasing':
        alphaRI = (T) / float(float(np.exp(beta * (T))) - 1)
        alphaRI = alphaRI
        phiVar = b0_D * alphaRI * (np.exp(beta * y0) - 1) + b1_D * alphaRI * (np.exp(beta * y1) - 1)
    else:
        print('Wrong input argument to additive fitness reward function')
        print('Argument must be linear, increasing or diminishing')
        exit(1)

    if argumentP == 'linear':
        psiVar = -(b0_D * y1 + b1_D * y0)

    elif argumentP == 'diminishing':
        alphaPD = (T) / float(1 - float(np.exp(-beta * (T))))
        alphaPD = alphaPD
        psiVar = -(b0_D * alphaPD * (1 - np.exp(-beta * y1)) + b1_D * alphaPD * (1 - np.exp(-beta * y0)))

    elif argumentP == 'increasing':
        alphaPI = (T) / float(float(np.exp(beta * (T))) - 1)
        alphaPI = alphaPI
        psiVar = -(b0_D * alphaPI * (np.exp(beta * y1) - 1) + b1_D * alphaPI * (np.exp(beta * y0) - 1))
    else:
        print('Wrong input argument to additive fitness penalty function')
        print('Argument must be linear, increasing or diminishing')
        exit(1)

    tf = 0 + phiVar + psi_weighting * psiVar

    return float(tf)


# terminal fitness function

def terminalFitness(state, posE0, posE1, argumentR, argumentP, T, beta, psi_weighting):
    """
    in the changing environments model we need to loop through the adult life span and calculate fitness at every time point
    """
    # transform the markov chain dictionary into a matrix
    b0_D = posE0
    b1_D = posE1
    fitnessVal = fitnessFunc(state, b0_D, b1_D, argumentR, argumentP, T, beta, psi_weighting)

    return fitnessVal


"""
- possible posteriors at the end of ontogeny won't change regardless of adult lifespan
- therefore store on tree-dictionary per time step 
"""


def buildPosteriorTree(x0Values, x1Values, pE0, pE1, pC0E0, pC1E0, T, tree_path):
    tValues = np.arange(T, -1, -1)
    for t in tValues:  # range of tValues is 0 till T-1, because we also want to store the prior
        posteriorTree = {}
        allCombinations = (itertools.product(*[x0Values, x1Values]))
        dataSets = list(((x0, x1) for x0, x1 in allCombinations if sum([x0, x1]) == t))
        pC1E1 = pC0E0
        pC0E1 = pC1E0
        for data in dataSets:
            x0, x1 = data
            pDE0 = binom.pmf(x0, x0 + x1, pC0E0)
            pDE1 = binom.pmf(x1, x0 + x1, pC1E1)
            b0_D, b1_D = BayesianUpdating(pE0, pE1, pDE0, pDE1)
            pC0D = pC0E0 * b0_D + pC0E1 * b1_D
            pC1D = pC1E0 * b0_D + pC1E1 * b1_D
            posteriorTree[data] = (tuple([float(b0_D), float(b1_D)]), tuple([float(pC0D), float(pC1D)]))
        pickle.dump(posteriorTree, open(os.path.join(tree_path, "tree%s.p" % (t)), 'wb'))
    del posteriorTree


def calcTerminalFitness(tree_path, y0Values, y1Values, y0Deconstruct, y1Deconstruct, ywValues, T, beta, argumentR,
                        argumentP,
                        psi_weightingParam):
    # if it doesn't exist yet create a directory for the final timestep
    trees = pickle.load(open(os.path.join(tree_path, "tree%s.p" % (T)), "rb"))
    print("Number of binary trees in the last step: " + str(len(trees)))
    terminalF = {}
    for D in trees.keys():
        x0, x1 = D
        posE0, posE1 = trees[D][0]
        # what are possible combinations of phenotypes?
        # this final combinations of phenotypes is the same even if there is a tradeoff btw
        # construction and deconstruction and even if they can deconstruct fully
        # if there is a trade-off between constructing and deconstructing all dimension do not necessarily sum to T
        # we could say they are smaller than or equal T or we add two additional phenotype dimensions:
        # E0+,E1- and E0-,E1+; these then sum to 1
        # if deconstruction takes place in one go I need to store an identifier indicating the last time period
        # during which deconstruction takes place. Yx then corresponds to YxC - ident
        allCombinations = (itertools.product(*[y0Values, y1Values, y0Deconstruct, y1Deconstruct, ywValues]))
        phenotypes = list(((y0C, y1C, y0D, y1D, yw) for y0C, y1C, y0D, y1D, yw in allCombinations if
                           (sum([y0C, y1C, y0D, y1D, yw]) == T) and (y0C >= y0D and y1C >= y1D)))

        for y0C, y1C, y0D, y1D, yw in phenotypes:

            y0 = y0C - y0D
            y1 = y1C - y1D
            currKey = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1, y0C, y1C, y0D, y1D, yw, T, y0,y1)

            # should fitness only depend on the number of actually made phenotypic adjustments?
            terminalF[currKey] = terminalFitness((x0, x1, y0, y1, yw), posE0,
                                                 posE1, argumentR,
                                                 argumentP, T, beta, psi_weightingParam)
    return terminalF


# @profile
def F(x0, x1, y0C, y1C, y0D, y1D, yw,y0,y1, posE0, posE1, pC1D, pC0D, t, argumentR, argumentP, T, finalTimeArr):
    """
    this is the core of SDP
    """

    posE0, posE1, pC1D, pC0D = float(posE0), float(posE1), float(pC1D), float(pC0D)
    # computes the optimal decision in each time step
    # computes the maximal expected fitness for decision made between time t and T for valid combinations
    # of environmental variables
    prFT = float(finalTimeArr[t + 1])
    # for generating agents
    # SECOND: calculate expected fitness at the end of lifetime associated with each of the developmental decisions
    if prFT != 0:
        TF = prFT * terminalFitness((x0, x1, y0, y1, yw), posE0, posE1,
                                    argumentR, argumentP, T)
    else:
        TF = 0

    t1 = t + 1

    if t == T - 1:  # is this is the last decision and we have seen all cues

        currKey1 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1, y0C + 1, y1C, y0D, y1D, yw, t1,y0+1,y1)
        currKey2 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1, y0C, y1C + 1, y0D, y1D, yw, t1,y0,y1+1)

        if y0C > y0D: #equivalent to saying that you cannoy deconstruct below 0
            currKey3 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1, y0C, y1C, y0D + 1, y1D, yw, t1,y0-1,y1)
        else:
            currKey3 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1, y0C, y1C, y0D, y1D, yw+1, t1,y0,y1)

        if y1C > y1D:
            currKey4 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1, y0C, y1C, y0D, y1D + 1, yw, t1,y0,y1-1)
        else:
            currKey4 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1, y0C, y1C, y0D, y1D, yw+1, t1, y0,y1)
        currKey5 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1, y0C, y1C, y0D, y1D, yw + 1, t1,y0,y1)

        F0C = Ft1[currKey1]
        F1C = Ft1[currKey2]
        F0D = Ft1[currKey3]
        F1D = Ft1[currKey4]
        Fw = Ft1[currKey5]

        maxF, maxDecision = findOptimumClose([F0C, F1C, F0D, F1D, Fw])

        # This code deals with the case in which deconstruction was not possible and the organism had to wait instead.
        # Under these conditions, we need recode the optimal decision to relfect the waiting.
        if (isinstance(maxDecision, tuple)):
            if 2 in maxDecision and int(currKey3.split(";")[6]) > yw:
                # find the index of 2
                maxDecision = list(maxDecision)
                i = maxDecision.index(2)
                maxDecision = maxDecision[:i] + [4] + maxDecision[i + 1:]
                maxDecision = tuple(set(tuple(maxDecision)))
                if len(maxDecision) ==1:
                    maxDecision = int(maxDecision[0])
        else:
            if 2 == maxDecision and int(currKey3.split(";")[6]) > yw:
                maxDecision = 4

        if (isinstance(maxDecision, tuple)):
            if 3 in maxDecision and int(currKey4.split(";")[6]) > yw:
                maxDecision = list(maxDecision)
                # find the index of 3
                i = maxDecision.index(3)
                maxDecision = maxDecision[:i] + [4] + maxDecision[i + 1:]
                maxDecision = tuple(set(tuple(maxDecision)))
                if len(maxDecision) ==1:
                    maxDecision = int(maxDecision[0])
        else:
            if 3 == maxDecision and int(currKey4.split(";")[6]) > yw:
                maxDecision = 4

    else:
        currKey1 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0 + 1, x1, y0C + 1, y1C, y0D, y1D, yw, t1,y0+1,y1)
        currKey2 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1 + 1, y0C + 1, y1C, y0D, y1D, yw, t1,y0+1,y1)

        currKey3 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0 + 1, x1, y0C, y1C + 1, y0D, y1D, yw, t1, y0,y1+1)
        currKey4 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1 + 1, y0C, y1C + 1, y0D, y1D, yw, t1, y0,y1+1)

        if y0C > y0D:
            currKey5 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0 + 1, x1, y0C, y1C, y0D + 1, y1D, yw, t1,y0-1,y1)
            currKey6 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1 + 1, y0C, y1C, y0D + 1, y1D, yw, t1,y0-1,y1)
        else:
            currKey5 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0 + 1, x1, y0C, y1C, y0D, y1D, yw+1, t1,y0,y1)
            currKey6 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1 + 1, y0C, y1C, y0D, y1D, yw+1, t1,y0,y1)

        if y1C > y1D:
            currKey7 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0 + 1, x1, y0C, y1C, y0D, y1D + 1, yw, t1, y0,y1-1)
            currKey8 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1 + 1, y0C, y1C, y0D, y1D + 1, yw, t1, y0,y1-1)
        else:
            currKey7 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0 + 1, x1, y0C, y1C, y0D, y1D, yw+1, t1,y0,y1)
            currKey8 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1 + 1, y0C, y1C, y0D, y1D, yw+1, t1,y0,y1)

        currKey9 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0 + 1, x1, y0C, y1C, y0D, y1D, yw + 1, t1,y0,y1)
        currKey10 = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1 + 1, y0C, y1C, y0D, y1D, yw + 1, t1,y0,y1)

        F0C = ((pC0D) * (Ft1[currKey1][0]) + (pC1D) * (Ft1[currKey2][0]))
        F1C = ((pC0D) * (Ft1[currKey3][0]) + (pC1D) * (Ft1[currKey4][0]))
        F0D = ((pC0D) * (Ft1[currKey5][0]) + (pC1D) * (Ft1[currKey6][0]))
        F1D = ((pC0D) * (Ft1[currKey7][0]) + (pC1D) * (Ft1[currKey8][0]))
        Fw = ((pC0D) * (Ft1[currKey9][0]) + (pC1D) * (Ft1[currKey10][0]))

        maxF, maxDecision = findOptimumClose([F0C, F1C, F0D, F1D, Fw])

        if (isinstance(maxDecision, tuple)):
            if 2 in maxDecision and int(currKey5.split(";")[6]) > yw:
                # find the index of 2
                maxDecision = list(maxDecision)
                i = maxDecision.index(2)
                maxDecision = maxDecision[:i] + [4] + maxDecision[i + 1:]
                maxDecision = tuple(set(tuple(maxDecision)))
                if len(maxDecision) ==1:
                    maxDecision = int(maxDecision[0])
        else:
            if 2 == maxDecision and int(currKey5.split(";")[6]) > yw:
                maxDecision = 4

        if (isinstance(maxDecision, tuple)):
            if 3 in maxDecision and int(currKey7.split(";")[6]) > yw:
                maxDecision = list(maxDecision)
                # find the index of 3
                i = maxDecision.index(3)
                maxDecision = maxDecision[:i] + [4] + maxDecision[i + 1:]
                maxDecision = tuple(set(tuple(maxDecision)))
                if len(maxDecision) ==1:
                    maxDecision = int(maxDecision[0])

        else:
            if 3 == maxDecision and int(currKey7.split(";")[6]) > yw:
                maxDecision = 4
    return float(maxF), maxDecision  # in order to track the current beliefs


def oneFitnessSweep(tree, t, phenotypes, finalTimeArr, T, argumentR, argumentP):
    # iterate through D in tree; for each D and phenotype combo store  the oprimal decision and accordingly
    # create a new fitness function but this paramerter can be stored later

    policyStar = {}
    for D in tree.keys():

        x0, x1 = D
        posE0, posE1 = tree[D][0]
        pC0D, pC1D = tree[D][1]

        for y0C, y1C, y0D, y1D, yw in phenotypes:
            y0 = y0C-y0D
            y1 = y1C-y1D
            fitness, optimalDecision = F(x0, x1, y0C, y1C, y0D, y1D, yw,y0,y1, posE0, posE1,
                                         pC1D, pC0D, t, argumentR, argumentP, T,
                                         finalTimeArr)

            currKey = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1, y0C, y1C, y0D, y1D, yw, t, y0,y1)
            policyStar[currKey] = (
                fitness, optimalDecision, pC1D, posE1)

    return policyStar


def printKeys(myShelve):
    for key in myShelve.keys()[0:10]:
        print(key)
        print(myShelve[key])


def run(priorE0, cueValidityC0E0, T, argumentR, argumentP, kwUncertainTime, beta,
        psi_weightingParam, pE0, pC0E0):
    # this is set to global so that it can be accessed from all parallel workers
    global Ft1
    Ft1 = {}
    treeLengthDict = {}

    # if it doesn't exist yet create a fitness directory
    if not os.path.exists('fitness'):
        os.makedirs('fitness')

    tree_path = 'trees'
    if not os.path.exists(tree_path):
        os.makedirs(tree_path)

    # the organism makes a decision in each of T-1 time periods to specialize towards P0, or P1 or wait
    T = int(T)

    # define input variable space

    # number of phnotypic adjustments towards either dimension
    y0Values = np.arange(0, T + 1, 1)
    y1Values = np.arange(0, T + 1, 1)
    # number of times deconstructed on either dimension
    y0Deconstruct = np.arange(0, T + 1, 1)
    y1Deconstruct = np.arange(0, T + 1, 1)
    # number of times waited
    ywValues = np.arange(0, T + 1, 1)

    x0Values = np.arange(0, T + 1, 1)
    x1Values = np.arange(0, T + 1, 1)

    tValues = np.arange(T - 1, -1, -1)
    tValuesForward = np.arange(1, T + 1, 1)

    pC1E0 = float(D(1) - D(pC0E0))

    """
    prepare the required parameters for the HMM algorithm 
    """
    pE1 = float(D(1) - D(pE0))

    # defines the random final time array
    # not really using this at the moment
    # should go back to this in the future
    finalTimeArrNormal = normData([probFinalTime(t, T, kwUncertainTime) for t in tValuesForward], probMinFT, probMaxFT)
    finalTimeArr = {t: v for t, v in zip(tValuesForward, finalTimeArrNormal)}
    pickle.dump(finalTimeArr, open('finalTimeArr.p', 'wb'))

    """""
    FORWARD PASS
    before calculating fitness values need to perform a forward pass in order to calculate the updated posterior values
    specifies a mapping from cue set combinations to posterior probabilities  
    
    """""
    if buildPosteriorTreeFlag:  # the posteriors only depend on the cue sequences and therefore should not need to be changed in this model
        startTime = time.perf_counter()
        print("start building tree")
        buildPosteriorTree(x0Values, x1Values, pE0, pE1, pC0E0, pC1E0, T, tree_path)
        elapsedTime = time.perf_counter() - startTime
        print('Elapsed time for computing the posterior tree: ' + str(elapsedTime))
    """""
    Calculate terminal fitness
    the next section section will call a function that uses the last terminal tree to calculate the fitness
    at the final time step
    """""
    print("calculate fitness for the last step")
    startTime = time.perf_counter()

    if not os.path.exists("fitness/%s" % T):
        os.makedirs("fitness/%s" % T)

    terminalFitness = calcTerminalFitness(tree_path, y0Values, y1Values, y0Deconstruct, y1Deconstruct,
                                          ywValues, T, beta, argumentR, argumentP,
                                          psi_weightingParam)
    # print(len(terminalFitness))
    pickle.dump(terminalFitness, open(os.path.join('fitness/%s' % T, "TF.p"), 'wb'))
    elapsedTime = time.perf_counter() - startTime
    print('Elapsed time for calculating terminal fitness: ' + str(elapsedTime))
    """""
    Stochastic dynamic programming via backwards induction 
    applies DP from T-1 to 1
    """""
    # developmental cycle; iterate from t = T-1 to t = 0 (backward induction)
    # TODO: Continue here
    for t in tValues:
        # IMPORTANT: note, that the t is referring to the sum of the phenotypic state variables, i.e., t+1 cues have been
        # sampled
        print("currently computing the optimal policy for time step %s" % t)

        t1 = t + 1

        if not os.path.exists("fitness/%s" % t):
            os.makedirs("fitness/%s" % t)

        # all possible combinations that can be achieved
        allCombinations = (itertools.product(*[y0Values, y1Values, y0Deconstruct, y1Deconstruct,
                                               ywValues]))
        phenotypes = list(((y0C, y1C, y0D, y1D, yw) for y0C, y1C, y0D, y1D, yw in allCombinations if
                           (sum([y0C, y1C, y0D, y1D, yw]) == t) and (y0C >= y0D and y1C >= y1D)))

        trees = pickle.load(open(os.path.join(tree_path, "tree%s.p" % (t1)), "rb"))
        # store the value in the treeLengthDict for plotting purposes
        treeLengthDict[t] = len(trees)

        if os.path.exists('fitness/%s/TF.p' % (t1)):
            currFt1 = pickle.load(open('fitness/%s/TF.p' % (t1), 'rb'))
            set_global_Ft1(currFt1)  # make the global fitness function available

        del currFt1

        # this is the bit that will do the actual fitness calculation

        """
        call the dynamic programming function here
        """
        currFt = oneFitnessSweep(trees, t, phenotypes, finalTimeArr, T, argumentR, argumentP)

        """
        store the current fitness / policy function
        """
        currentFile = os.path.join('fitness/%s' % t, "TF.p")
        pickle.dump(currFt, open(currentFile, 'wb'))

        set_global_Ft1({})
        del currFt
    del Ft1


def find_nearest(array, value):
    n = [abs(i - value) for i in array]
    idx = n.index(min(n))
    return idx


if __name__ == '__main__':
    # specify parameters

    """
    T is an integer specifying the number of discrete time steps of the model
    T is the only parameter read in via the terminal
    to run the model, type the following in the terminal: python VarCueRelModel.py T
    """

    T = sys.argv[1]

    optimalPolicyFlag = False  # would you like to compute the optimal policy
    buildPosteriorTreeFlag = False  # would you like compute posterior probabilities in HMM for all possible cue sets
    preparePlottingFlag = False # would you like to prepare the policy for plotting?
    plotFlag = True  # would you like to plot the aggregated data?
    standardPlots = True
    performSimulation = False  # do you need to simulate data based on the "preapredPlotting data" for plotting?; i.e.,
    # simulate twins and/or mature phenotypes
    # this is the directory where modeling results are stored
    #mainPath = "/home/nicole/Projects/Results_reversible_development_model/newResults/"
    mainPath = '/media/nicole/Elements/Results_reversible_development_model/newResults/'

    #mainPath = '/media/nicole/Elements/Results_reversible_development_model/10_ts/'

    if not os.path.exists(mainPath):
        os.makedirs(mainPath)

    # setting a prior (or an array of priors in case we want to do multiple runs at once)
    priorE0Arr = [0.5, 0.3, 0.1]

    # corresponds to the probability of receiving C0 when in E0
    cueValidityC0E0Arr = [0.55, 0.75, 0.95]

    argumentRArr = ['linear','diminishing','increasing'] #'linear','diminishing','increasing'
    argumentPArr = ['linear','diminishing','increasing']  # 'linear','diminishing','increasing'

    deconstruction = 'incremental'  # "all"
    tradeOffConstruction = True  # tradeoff between construction and deconstruction

    # parameters for the some of the internal functions (specifically the phenotype to fitness mapping),
    # change for specifc adaptations
    beta = 0.2  # beta determines the curvature of the fitness and reward functions
    # for uncertain time == step
    a = 10
    b = 18

    # for uncertain time == function:
    # define your own function
    kwUncertainTime = 'None'
    prob = 0.8  # the probability of dying after t = b
    funFT = lambda x: 0.5 * np.cos(x)  # (1-(x/float(21)))
    probMax = 0.5  # this will define the probability range for reaching the terminal state
    probMin = 0.0

    # penalty weighting
    psi_weightingParam = 1

    # global policy dict
    set_global_variables(a, b, prob, funFT, probMax, probMin)

    """
    ADD DOCUMENTATION
    """
    for argumentR in argumentRArr:
        for argumentP in argumentPArr:

            for priorE0 in priorE0Arr:
                for cueValidityC0E0 in cueValidityC0E0Arr:

                    print("Run with prior " + str(priorE0) + " and cue validity " + str(
                        cueValidityC0E0))
                    print("Run with reward " + str(argumentR) + " and penalty " + str(argumentP))

                    # create a folder for that specific combination, also encode the reward and penalty argument
                    currPath = "runTest_%s%s_%s%s" % (argumentR[0], argumentP[0], priorE0, cueValidityC0E0)

                    if not os.path.exists(os.path.join(mainPath, currPath)):
                        os.makedirs(os.path.join(mainPath, currPath))
                    # set the working directory for this particular parameter combination
                    os.chdir(os.path.join(mainPath, currPath))
                    
                    
                    if optimalPolicyFlag:
                        # calculates the optimal policy
                        run(priorE0, cueValidityC0E0, T, argumentR, argumentP, kwUncertainTime, beta,
                            psi_weightingParam, priorE0, cueValidityC0E0)
                    # works until here
                    if preparePlottingFlag:
                        print("prepare plotting")
                        T = int(T)
                        # probability of dying at any given state
                        finalTimeArr = pickle.load(open("finalTimeArr.p", "rb"))
                        pE1 = float(D(1) - D(priorE0))

                        # create plotting folder
                        if not os.path.exists('plotting'):
                            os.makedirs('plotting')

                        """
                        run code that prepares the optimal policy data for plotting
                        preparePlotting will prepare the results from the optimal policy for plotting and store them in the
                        folder for each parameter combination
                        """
                        preparePlotting(T, priorE0, pE1, kwUncertainTime,
                                        finalTimeArr)
                        print('Creating data for plots')

                        dataPath2 = 'plotting/aggregatedResults'
                        resultsPath = 'plotting/resultDataFrames'
                        os.chdir(os.path.join(mainPath, currPath))

                        joinIndidividualResultFiles('raw', np.arange(1, T + 1, 1), resultsPath)  # raw results
                        joinIndidividualResultFiles('aggregated', np.arange(1, T + 1, 1), dataPath2)  # raw results
                        joinIndidividualResultFiles('plotting', np.arange(1, T + 1, 1), dataPath2)  # raw results
                        """
                        PERHAPS CONSIDER DELETEING AT THIS POINT
                        Deleting large data files
                        - first from HDD location
                        - then SSD
                        """
                        # # can delete he complete folder on the HDD #TODO do I still need to delete things?
                        # os.chdir(mainPath)
                        # shutil.rmtree(currPath, ignore_errors=True)
                        # # next delete stuff from SSD
                        # shutil.rmtree(os.path.join(currPath, 'fitness'), ignore_errors=True)
                        # shutil.rmtree(os.path.join(currPath, 'plotting/StateDistribution'), ignore_errors=True)
                        # # empty the trash
                        # os.system('trash-empty')

    """
    Creating plots
    """
    if plotFlag:
        # what kind of plots can we make:
        # 1. policy plot
        # 2. rank order stability
        # 3. twinstudy and experimental twin study with all treatments and lags
        # 4. mature phenotypes and fitness differences
        # possible plotting arguments: Twinstudy, ExperimentalTwinstudy, MaturePhenotypes, FitnessDifference
        dataPath2 = 'plotting/aggregatedResults'
        for argumentR in argumentRArr:
            for argumentP in argumentPArr:
                os.chdir(mainPath)
                print("Plot for reward " + str(argumentR) + " and penalty " + str(argumentP))
                twinResultsPath = "PlottingResults_%s_%s" % (argumentR[0], argumentP[0])
                if not os.path.exists(twinResultsPath):
                    os.makedirs(twinResultsPath)
                r = 0.5#0.3,0.2  # this is the radius baseline for the policy plots np.sqrt(1/float(np.pi))#0.5
                minProb = 0  # minimal probability of reaching a state for those that are displayed

                numAgents = 10000
                baselineFitness = 0
                lag = [3]  # number of discrete time steps that twins are separated
                endOfExposure = False
                adoptionType = "yoked"
                plotVar = False  # do you want to plot variance in phenotypic distances?
                startEnvList = [1]  # ,1]  #1 specify the starting environment; normally the more likely environment

                if standardPlots:
                    for startEnv in startEnvList:

                        # policyPlotReduced_v2(int(T) + 1, r, priorE0Arr, cueValidityC0E0Arr,
                        #                  np.arange(1, int(T) + 1, 1), dataPath2, True,
                        #                  argumentR, argumentP, minProb,
                        #                  mainPath, twinResultsPath)
                        #
                        # os.chdir(mainPath)

                        policyPlotReduced(int(T) + 1, r, priorE0Arr, cueValidityC0E0Arr,
                                         np.arange(1, int(T) + 1, 1), dataPath2, True,
                                         argumentR, argumentP, minProb,
                                         mainPath, twinResultsPath)

                        os.chdir(mainPath)

                        '''
                        The subsequent code block is specifically for variants of adoption studies
                        '''
                        # plotArgs = ["Twinstudy"]  # "Twinstudy","BeliefTwinstudy", "ExperimentalTwinstudy"]
                        # for adoptionType in ["yoked"]: #, "oppositePatch","deprivation"
                        #     print("adoptionType: " + str(adoptionType))
                        #     runPlots(priorE0Arr, cueValidityC0E0Arr, int(T), numAgents,
                        #              twinResultsPath,
                        #              baselineFitness,
                        #              mainPath, argumentR,
                        #              argumentP, lag, adoptionType, False, plotArgs, plotVar,
                        #              performSimulation,
                        #              startEnv)

                        # plotArgs = ["ExperimentalTwinstudy"]
                        # for adoptionType in ["yoked"]:  # , "oppositePatch","deprivation"]:
                        #     print("adoptionType: " + str(adoptionType))
                        #     runPlots(priorE0Arr, cueValidityC0E0Arr, int(T), numAgents,
                        #              twinResultsPath,
                        #              baselineFitness,
                        #              mainPath, argumentR,
                        #              argumentP, lag, adoptionType, False, plotArgs, plotVar,
                        #              performSimulation,
                        #              startEnv)
                        #
                        #     #with end of exposure set to True
                        #     runPlots(priorE0Arr, cueValidityC0E0Arr, int(T), numAgents,
                        #              twinResultsPath,
                        #              baselineFitness,
                        #              mainPath, argumentR,
                        #              argumentP, lag, adoptionType, True, plotArgs, plotVar,
                        #              performSimulation,
                        #              startEnv)

                        # plotArgs = ["MaturePhenotypes", "FitnessDifference"]  #"RankOrderStability" "FitnessDifference", "MaturePhenotypes",
                        # runPlots(priorE0Arr, cueValidityC0E0Arr, int(T), numAgents, twinResultsPath,
                        #          baselineFitness,
                        #          mainPath, argumentR, argumentP, lag, adoptionType, endOfExposure, plotArgs,
                        #          plotVar,
                        #          performSimulation,startEnv)


