import numpy as np
import os
import pandas as pd
import time
from multiprocessing import Pool
import itertools
import math
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.markers import MarkerStyle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch, ArrowStyle
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

"""""
This script will run experimental twin studies to test how sensitive different optimal policies are to cues
    in these experiments I simulate agents in specific environments based on the previously calculated optimal policies
    importantly I will try to use the finalRAW file rather than the dictionary files 
    
    procedure:
    - simulate twins who are identical up to time period t 
    - keep one twin ("original") in its natal patch
    - send the other twin ("doppelgaenger") to mirror patch 
    - doppelgaenger receives opposite (yoked) cues from the original twin
        the cues are opposite but not from the opposite patch
"""""




# class Arrow3D(FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         super().__init__((0,0), (0,0), *args, **kwargs)
#         self._verts3d = xs, ys, zs
#
#     def do_3d_projection(self, renderer=None):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
#         self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
#
#         return np.min(zs)

def setGlobalPolicy(policyPath):
    global policy
    policy = pd.read_csv(policyPath, index_col=0).reset_index(drop=True)


def func_star(allArgs):
    return simulateTwins(*allArgs)


def func_star2(allArgs):
    return simulateExperimentalTwins(*allArgs)


def updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker):
    optDecisions = [subDF.loc[(subDF['x0'] == cueTracker[idx, 0]) & (subDF['x1'] == cueTracker[idx, 1]) & (
            subDF['y0C'] == phenotypeTracker[idx, 0]) & (subDF['y1C'] == phenotypeTracker[idx, 1]) & (
                                      subDF['y0D'] == phenotypeTracker[idx, 2]) & (
                                      subDF['y1D'] == phenotypeTracker[idx, 3]) & (
                                      subDF['yw'] == phenotypeTracker[idx, 4]) & (
                                      subDF['y0'] == phenotypeTracker[idx, 5]) & (
                                      subDF['y1'] == phenotypeTracker[idx, 6])]['cStar'].item() for idx in
                    simValues]

    # additionally keep track of the posterior belief
    posBelief = [subDF.loc[(subDF['x0'] == cueTracker[idx, 0]) & (subDF['x1'] == cueTracker[idx, 1]) & (
            subDF['y0C'] == phenotypeTracker[idx, 0]) & (subDF['y1C'] == phenotypeTracker[idx, 1]) & (
                                   subDF['y0D'] == phenotypeTracker[idx, 2]) & (
                                   subDF['y1D'] == phenotypeTracker[idx, 3]) & (
                                   subDF['yw'] == phenotypeTracker[idx, 4]) & (
                                   subDF['y0'] == phenotypeTracker[idx, 5]) & (
                                   subDF['y1'] == phenotypeTracker[idx, 6])]['pE1'].item() for idx in
                 simValues]
    # post process optimal decisions
    optDecisionsNum = [
        int(a) if not '(' in str(a) else int(np.random.choice(str(a).replace("(", "").replace(")", "").split(","))) for
        a in
        optDecisions]

    # update phenotype tracker
    idx0 = [idx for idx, val in enumerate(optDecisionsNum) if val == 0]
    if len(idx0) > 0:
        phenotypeTracker[idx0, 0] += 1  # construction y0
        phenotypeTracker[idx0, 5] += 1  # construction y0

    idx1 = [idx for idx, val in enumerate(optDecisionsNum) if val == 1]
    if len(idx1) > 0:
        phenotypeTracker[idx1, 1] += 1  # construction y1
        phenotypeTracker[idx1, 6] += 1  # construction y0

    idx2 = [idx for idx, val in enumerate(optDecisionsNum) if val == 2]
    if len(idx2) > 0:
        idx2_Dpossible = [idx for idx in idx2 if phenotypeTracker[idx, 0] > phenotypeTracker[idx, 2]]
        idx2_Dimpossible = [idx for idx in idx2 if phenotypeTracker[idx, 0] <= phenotypeTracker[idx, 2]]
        phenotypeTracker[idx2_Dpossible, 2] += 1  # deconstruction y0
        phenotypeTracker[idx2_Dpossible, 5] -= 1
        phenotypeTracker[idx2_Dimpossible, 4] += 1  # if we cannot deconstruct, we wait

    idx3 = [idx for idx, val in enumerate(optDecisionsNum) if val == 3]
    if len(idx3) > 0:
        idx3_Dpossible = [idx for idx in idx3 if phenotypeTracker[idx, 1] > phenotypeTracker[idx, 3]]
        idx3_Dimpossible = [idx for idx in idx3 if phenotypeTracker[idx, 1] <= phenotypeTracker[idx, 3]]

        phenotypeTracker[idx3_Dpossible, 3] += 1  # deconstruction y1
        phenotypeTracker[idx3_Dpossible, 6] -= 1
        phenotypeTracker[idx3_Dimpossible, 4] += 1  # if we cannot deconstruct, we wait

    idx4 = [idx for idx, val in enumerate(optDecisionsNum) if val == 4]
    if len(idx4) > 0:
        phenotypeTracker[idx4, 4] += 1  # waiting

    return phenotypeTracker, posBelief


def simulateExperimentalTwins(tAdopt, twinNum, env, cueReliability, lag, T, adoptionType, endOfExposure):
    """
    This function is smulating twins following the optimal policy up until time point t
    after t one twin receives yoked opposite cues

    pE1 is the prior probability of being in environment 1
    pc1E1 is the cue reliability
    :return: phenotypic distance between pairs of twins
    """
    T = T + lag - 1
    tValues = np.arange(1, tAdopt, 1)

    if env == 1:
        pC1Start = cueReliability  # take the very first cue reliability
    else:
        pC1Start = 1 - cueReliability
    pC0Start = 1 - pC1Start

    if isinstance(twinNum, list):
        allCues = np.stack(twinNum)
        twinNum = len(twinNum)

    else:
        # shape agents x time
        allCues = np.stack(list(np.random.choice([0, 1], size=T, p=[pC0Start, pC1Start]) for x in np.arange(twinNum)))
    # determine the probabilities of the sequences of cues
    cueProbabilities = []
    cueProbabilities += [calcSequenceProb(cues, env, cueReliability) for cues in allCues]

    # simulate the first pair of twins
    cues = [int(cue) for cue in allCues[:, 0]]
    cues = np.array(cues)

    # need to reverse the last update
    if adoptionType == "yoked":
        oppositeCues = 1 - cues
    elif adoptionType == "oppositePatch":
        oppositeCues = np.random.choice([0, 1], size=twinNum, p=[pC1Start, pC0Start])
        oppositeCues = np.array(oppositeCues)
    elif adoptionType == "deprivation":
        oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
        oppositeCues = np.array(oppositeCues)

    else:
        print("wrong input argument to adoption type!")
        exit(1)

    cueTracker = np.zeros((twinNum, 2))
    cueTracker[:, 0] = 1 - cues
    cueTracker[:, 1] = cues
    phenotypeTracker = np.zeros((twinNum, 7))  # y0C,y1C,y0D,y1D,yw,y0,y1

    simValues = np.arange(0, twinNum, 1)

    for t in tValues:
        subDF = policy[policy['time'] == t].reset_index(drop=True)

        phenotypeTracker, __ = updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker)
        if t < T:
            # next cue sequences
            cues = [int(cue) for cue in allCues[:, t]]
            cues = np.array(cues)
            # need to reverse the last update
            if adoptionType == "yoked":
                oppositeCues = 1 - cues
            elif adoptionType == "oppositePatch":
                oppositeCues = np.random.choice([0, 1], size=twinNum, p=[pC1Start, pC0Start])
                oppositeCues = np.array(oppositeCues)
            elif adoptionType == "deprivation":
                oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
                oppositeCues = np.array(oppositeCues)

        # update cue tracker
        cueTracker[:, 0] += (1 - cues)
        cueTracker[:, 1] += cues

    # post adoption period
    # continue here
    originalTwin = np.copy(phenotypeTracker)
    doppelgaenger = np.copy(phenotypeTracker)

    restPeriod = np.arange(tAdopt, tAdopt + lag, 1)

    # setting up the matrix for the yoked opposite cues
    cueTrackerDoppel = np.copy(cueTracker)

    cueTrackerDoppel[:, 0] += -(1 - cues) + (1 - oppositeCues)
    cueTrackerDoppel[:, 1] += -cues + oppositeCues

    for t2 in restPeriod:

        np.random.seed()
        subDF = policy[policy['time'] == t2].reset_index(drop=True)

        # update the phenotypes of the twins
        originalTwin, __ = updatePhenotype(subDF, originalTwin, simValues, cueTracker)

        doppelgaenger, __ = updatePhenotype(subDF, doppelgaenger, simValues, cueTrackerDoppel)
        if t2 < T:

            cuesOriginal = [int(cue) for cue in allCues[:, t2]]
            cuesOriginal = np.array(cuesOriginal)

            if adoptionType == "yoked":
                oppositeCues = 1 - cuesOriginal
            elif adoptionType == "oppositePatch":
                oppositeCues = np.random.choice([0, 1], size=twinNum, p=[pC1Start, pC0Start])
                oppositeCues = np.array(oppositeCues)
            else:  # adoptionType = deprivation
                oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
                oppositeCues = np.array(oppositeCues)

        # update cue tracker
        cueTracker[:, 0] += (1 - cuesOriginal)
        cueTracker[:, 1] += cuesOriginal

        cueTrackerDoppel[:, 0] += (1 - oppositeCues)
        cueTrackerDoppel[:, 1] += oppositeCues

    restPeriodReunited = np.arange(tAdopt + lag, T + 1, 1)
    # need to reverse the last update
    cueTrackerDoppel[:, 0] += -(1 - oppositeCues) + (1 - cuesOriginal)
    cueTrackerDoppel[:, 1] += -(oppositeCues) + cuesOriginal

    if not endOfExposure:  # this means we want to measure phenotypic distance at the end of onotgeny
        for t3 in restPeriodReunited:
            # they will receive the same cues again
            np.random.seed()
            # print "currently simulating time step: " + str(t)
            subDF = policy[policy['time'] == t3].reset_index(drop=True)

            originalTwin, __ = updatePhenotype(subDF, originalTwin, simValues, cueTracker)
            doppelgaenger, __ = updatePhenotype(subDF, doppelgaenger, simValues, cueTrackerDoppel)

            if t3 < T:
                cuesOriginal = [int(cue) for cue in allCues[:, t2]]
                cuesOriginal = np.array(cuesOriginal)

            # update cue tracker
            cueTracker[:, 0] += (1 - cuesOriginal)
            cueTracker[:, 1] += cuesOriginal

            cueTrackerDoppel[:, 0] += (1 - cuesOriginal)
            cueTrackerDoppel[:, 1] += cuesOriginal

    return originalTwin, doppelgaenger, cueProbabilities


def calcSequenceProb(cues, env, cueVal):
    if env == 0:
        pE0 = 1
        pE1 = 0
    else:
        pE0 = 0
        pE1 = 1
    cues = [int(cue) for cue in cues]
    x1 = sum(cues)
    x0 = len(cues) - x1
    pDE0 = binom.pmf(x0, x0 + x1, float(cueVal))
    pDE1 = binom.pmf(x1, x0 + x1, float(cueVal))
    p_D = pDE0 * float(pE0) + pDE1 * float(pE1)
    # how often can this total combination of cues occur?
    total = scipy.special.comb(x0 + x1, x0)

    return float(p_D) / float(total)


def simulateTwins(tAdopt, twinNum, env, adopt, T, adoptionType, pE0, cueVal):
    """
    This function is smulating twins following the optimal policy up until time point t
    after t one twin receives yoked opposite cues

    pE1 is the prior probability of being in environment 1
    pc1E1 is the cue reliability array!
    :return: phenotypic distance between pairs of twins
    """

    """
    compute an exact behavior for T <= 9 
    """
    if env == 1:
        pC1Env = cueVal
    else:
        pC1Env = 1 - cueVal
    pC0Env = 1 - pC1Env

    if isinstance(twinNum, list):
        allCues = np.stack(twinNum)
        twinNum = len(twinNum)

    else:
        # shape agents x time
        allCues = np.stack(list(np.random.choice([0, 1], size=T, p=[pC0Env, pC1Env]) for x in np.arange(twinNum)))

    # determine the probabilities of the sequences of cues
    cueProbabilities = []
    cueProbabilities += [calcSequenceProb(cues, env, cueVal) for cues in allCues]

    if adopt:
        tValues = np.arange(1, tAdopt, 1)
        cues = [int(cue) for cue in allCues[:, 0]]
        cues = np.array(cues)

        if adoptionType == "yoked":
            oppositeCues = 1 - cues
        elif adoptionType == "oppositePatch":
            oppositeCues = np.random.choice([0, 1], size=twinNum, p=[pC1Env, pC0Env])
            oppositeCues = np.array(oppositeCues)
        elif adoptionType == "deprivation":
            oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
            oppositeCues = np.array(oppositeCues)
        else:
            print("wrong input argument to adoption type!")
            exit(1)

        cueTracker = np.zeros((twinNum, 2))
        cueTracker[:, 0] = 1 - cues
        cueTracker[:, 1] = cues
        phenotypeTracker = np.zeros((twinNum, 7))  # y0C,y1C,y0D,y1D,yw,y0,y1
        posBeliefTracker = [0] * twinNum

        simValues = np.arange(0, twinNum, 1)

        for t in tValues:

            np.random.seed()
            # print "currently simulating time step: " + str(t)
            subDF = policy[policy['time'] == t].reset_index(drop=True)
            # next generate 10000 new cues
            # generate 10000 optimal decisions

            # TODO this is the parts that needs adjustment; we need to store multiple phenotypic dimensions now
            phenotypeTracker, posBeliefTracker = updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker)
            if t < T:
                cues = [int(cue) for cue in allCues[:, t]]
                cues = np.array(cues)

                if adoptionType == "yoked":
                    oppositeCues = 1 - cues
                elif adoptionType == "oppositePatch":
                    oppositeCues = np.random.choice([0, 1], size=twinNum, p=[pC1Env, pC0Env])
                    oppositeCues = np.array(oppositeCues)
                else:  # adoptionType = deprivation
                    oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
                    oppositeCues = np.array(oppositeCues)

            # update cue tracker
            cueTracker[:, 0] += (1 - cues)
            cueTracker[:, 1] += cues

        # post adoption period
        # continue here
        originalTwin = np.copy(phenotypeTracker)
        doppelgaenger = np.copy(phenotypeTracker)

        posBeliefTrackerOrg = np.zeros((twinNum, T + 1 - tAdopt + 1))
        posBeliefTrackerDG = np.zeros((twinNum, T + 1 - tAdopt + 1))

        # for the first time point where twins are separated the whole time we only add a placeholder for the prior
        # an array of zeros; therefore the postprocessinf needs to be doner atfer the arguments have been returned
        posBeliefTrackerOrg[:, 0] = posBeliefTracker
        posBeliefTrackerDG[:, 0] = posBeliefTracker
        del posBeliefTracker

        restPeriod = np.arange(tAdopt, T + 1, 1)

        # setting up the matrix for the yoked opposite cues
        cueTrackerDoppel = np.copy(cueTracker)

        cueTrackerDoppel[:, 0] += -(1 - cues) + (1 - oppositeCues)
        cueTrackerDoppel[:, 1] += -cues + oppositeCues

        for t2 in restPeriod:  # this is where adoption starts
            np.random.seed()
            subDF = policy[policy['time'] == t2].reset_index(drop=True)

            # update the phenotypes of the twins
            originalTwin, posBeliefOrg = updatePhenotype(subDF, originalTwin, simValues, cueTracker)

            posBeliefTrackerOrg[:, t2 - tAdopt + 1] = posBeliefOrg
            doppelgaenger, posBeliefDG = updatePhenotype(subDF, doppelgaenger, simValues, cueTrackerDoppel)
            posBeliefTrackerDG[:, t2 - tAdopt + 1] = posBeliefDG

            if t2 < T:
                cuesOriginal = [int(cue) for cue in allCues[:, t2]]
                cuesOriginal = np.array(cuesOriginal)

                if adoptionType == "yoked":
                    oppositeCues = 1 - cuesOriginal
                elif adoptionType == "oppositePatch":
                    oppositeCues = np.random.choice([0, 1], size=twinNum, p=[pC1Env, pC0Env])
                    oppositeCues = np.array(oppositeCues)
                else:  # adoptionType = deprivation
                    oppositeCues = np.random.choice([0, 1], size=twinNum, p=[0.5, 0.5])
                    oppositeCues = np.array(oppositeCues)

                # update cue tracker
                cueTracker[:, 0] += (1 - cuesOriginal)
                cueTracker[:, 1] += cuesOriginal

                cueTrackerDoppel[:, 0] += (1 - oppositeCues)
                cueTrackerDoppel[:, 1] += oppositeCues

            # store the very first phenotype following adotption to limit the amount of data you need to store
            if t2 == tAdopt:
                originalTwinTemp = np.copy(originalTwin)
                doppelgaengerTemp = np.copy(doppelgaenger)

        return originalTwin, doppelgaenger, posBeliefTrackerOrg, posBeliefTrackerDG, originalTwinTemp, doppelgaengerTemp, cueProbabilities


    else:  # to just calculate mature phenotypes and rank order stability

        tValues = np.arange(1, T + 1, 1)

        cuesSTart = [int(cue) for cue in allCues[:, 0]]
        cuesSTart = np.array(cuesSTart)
        cueTracker = np.zeros((twinNum, 2))

        cueTracker[:, 0] = 1 - cuesSTart
        cueTracker[:, 1] = cuesSTart
        phenotypeTracker = np.zeros((twinNum, 7))
        phenotypeTrackerTemporal = np.zeros((twinNum, 7, T))
        posBeliefTrackerTemporal = np.zeros((twinNum, T))

        simValues = np.arange(0, twinNum, 1)
        for t in tValues:

            np.random.seed()
            subDF = policy[policy['time'] == t].reset_index(drop=True)

            # print identTracker
            phenotypeTracker, posBelief = updatePhenotype(subDF, phenotypeTracker, simValues, cueTracker)
            phenotypeTrackerTemporal[:, :, t - 1] = np.copy(phenotypeTracker)
            posBeliefTrackerTemporal[:, t - 1] = np.copy(posBelief)

            if t < T:
                # now we have to recompute this for every timestep
                cues = [int(cue) for cue in allCues[:, t]]
                cues = np.array(cues)
                # update cue tracker
                cueTracker[:, 0] += (1 - cues)
                cueTracker[:, 1] += cues

        # successfully computed mature phenotypes
        return phenotypeTracker, phenotypeTrackerTemporal, posBeliefTrackerTemporal, cueProbabilities


def runExperimentalTwinStudiesParallel(tAdopt, twinNum, env, pE0, pC1E1, lag, T, resultsPath, argumentR,
                                       argumentP,
                                       adoptionType, endOfExposure):
    policyPath = os.path.join(resultsPath,
                              'runTest_%s%s_%s%s/finalRaw.csv' % (argumentR[0], argumentP[0], pE0, pC1E1))
    setGlobalPolicy(policyPath)

    # load the cue reliability array
    pC1E1 = D(str(pC1E1))
    pE0 = D(str(pE0))
    TLag = T + lag - 1
    allCues, probabilities = generateCueSequencesAndProabilities(TLag, env, pC1E1)

    if len(probabilities) <= twinNum:
        simulationChunk = chunks(allCues,
                                 12)  # thi provides sublists of length 12 each, not exactly what I wanted bu perhaps it works
    else:
        simulationChunk = [int(math.ceil(float(twinNum) / 12))] * 12

    pool = Pool(processes=12)

    results = pool.map(func_star2, zip(itertools.repeat(tAdopt),
                                       simulationChunk, itertools.repeat(env), itertools.repeat(pC1E1),
                                       itertools.repeat(lag),
                                       itertools.repeat(T), itertools.repeat(adoptionType),
                                       itertools.repeat(endOfExposure)
                                       ))
    pool.close()
    pool.join()

    results1a, results2a, results3a = zip(*results)
    if len(probabilities) > twinNum:
        return np.concatenate(results1a), np.concatenate(results2a), None
    else:
        return np.concatenate(results1a), np.concatenate(results2a), np.concatenate(results3a)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def generateCueSequencesAndProabilities(T, env, pC1E1):
    if env == 0:
        pE0 = 1
        pE1 = 0
    else:
        pE0 = 0
        pE1 = 1

    x0Values = np.arange(0, T + 1, 1)
    x1Values = np.arange(0, T + 1, 1)
    cueSequences = []
    cueProbabilities = []
    for x0 in x0Values:
        for x1 in x1Values:
            if x0 + x1 == T:
                current = list(multiset_permutations(['0'] * x0 + ['1'] * x1))
                pDE0 = binom.pmf(x0, x0 + x1, float(pC1E1))
                pDE1 = binom.pmf(x1, x0 + x1, float(pC1E1))
                p_D = pDE0 * float(pE0) + pDE1 * float(pE1)

                cueProbabilities += [float(p_D / len(current))] * len(current)
                cueSequences += current

    return cueSequences, cueProbabilities


def find_nearest(array, value):
    n = [abs(i - value) for i in array]
    idx = n.index(min(n))
    return idx


def runTwinStudiesParallel(tAdopt, twinNum, env, pE0, pC1E1, adopt, T, resultsPath, argumentR, argumentP, adoptionType):
    policyPath = os.path.join(resultsPath,
                              'runTest_%s%s_%s%s/finalRaw.csv' % (argumentR[0], argumentP[0], pE0, pC1E1))
    setGlobalPolicy(policyPath)

    # load the cue reliability array
    pC1E1 = D(str(pC1E1))
    pE0 = D(str(pE0))

    allCues, probabilities = generateCueSequencesAndProabilities(T, env, pC1E1)

    if len(probabilities) <= twinNum:
        simulationChunk = chunks(allCues,
                                 12)  # this provides sublists of length 12 each, not exactly what I wanted bu perhaps it works

    else:
        simulationChunk = [int(math.ceil(float(twinNum) / 12))] * 12

    pool = Pool(processes=12)
    if adopt:

        results = pool.map(func_star, zip(itertools.repeat(tAdopt),
                                          simulationChunk, itertools.repeat(env),
                                          itertools.repeat(adopt),
                                          itertools.repeat(T), itertools.repeat(adoptionType),
                                          itertools.repeat(pE0),
                                          itertools.repeat(pC1E1)))
        pool.close()
        pool.join()

        # results1, results2 refer to the phenotypes of orginals and clones
        # results3, results4 refer to the belief matrices of original and clone; shape: numAgents x separationTime +1
        results1a, results2a, results3a, results4a, results5a, results6a, result7a = zip(*results)

        if len(probabilities) > twinNum:
            return np.concatenate(results1a), np.concatenate(results2a), np.concatenate(results3a), np.concatenate(
                results4a), np.concatenate(results5a), np.concatenate(results6a), None
        else:
            return np.concatenate(results1a), np.concatenate(results2a), np.concatenate(results3a), np.concatenate(
                results4a), np.concatenate(results5a), np.concatenate(results6a), np.concatenate(result7a)

    else:

        pool = Pool(processes=12)

        results = pool.map(func_star, zip(itertools.repeat(tAdopt),
                                          simulationChunk, itertools.repeat(env),
                                          itertools.repeat(adopt),
                                          itertools.repeat(T), itertools.repeat(adoptionType),
                                          itertools.repeat(pE0),
                                          itertools.repeat(pC1E1)))
        pool.close()
        pool.join()

        results1a, results2a, results3a, results4a = zip(*results)

        if len(probabilities) > twinNum:
            return np.concatenate(results1a), np.concatenate(results2a), np.concatenate(results3a), None
        else:
            return np.concatenate(results1a), np.concatenate(results2a), np.concatenate(results3a), np.concatenate(
                results4a)


def calcEuclideanDistance(original, doppelgaenger, idxTuple):
    idx1 = idxTuple[0]
    idx2 = idxTuple[1] + 1
    result = [np.sqrt(np.sum((x - y) ** 2)) for x, y in zip(original[:, idx1:idx2], doppelgaenger[:, idx1:idx2])]
    return np.array(result)


def runExperimentalAdoptionExperiment(T, numAgents, env, prior, cueReliability, resultsPath, argumentR, argumentP, lag,
                                      adoptionType, endOfExposure):
    # this function will run twinstudies for a specific parameter combination for each possible moment of adoption

    # absolute phenotypic distance: average distance between numAgents organisms and their doppelgaengers at the end
    # of development

    # relative distance: absolute distance divided by maximum possible distance
    # maximum possible distance: T * sqrt(2)
    tValues = np.arange(1, T + 1, 1)
    TLag = T + lag - 1
    resultLen = determineLength(TLag)
    if resultLen > numAgents:
        resultLen = int(math.ceil(float(numAgents) / 12)) * 12

    results = np.zeros((T, resultLen))

    # new dictionaries for storing the different types of phenotypic distances
    # we are computing phenotypic distances between y0 and y1
    # between y0C and y1C
    # and between y0D and y1D

    allResults = {}
    allResultsTemp = {}
    namedList = {'y0y1': (5, 6), 'y0Cy1C': (0, 1), 'y0Dy1D': (2, 3)}
    for key, val in namedList.items():
        allResults[key] = results.copy()

    atypicalTraces = {}
    typicalTraces = {}

    for t in tValues:
        print("currently working on time step: " + str(t))
        original, doppelgaenger, cueProbabilities = runExperimentalTwinStudiesParallel(t, numAgents, env, prior,
                                                                                       cueReliability, lag, T,
                                                                                       resultsPath, argumentR,
                                                                                       argumentP, adoptionType,
                                                                                       endOfExposure)

        if isinstance(cueProbabilities, np.ndarray):
            cueProbabilities = [float(elem) for elem in cueProbabilities]

        for key, val in namedList.items():
            allResults[key][t - 1, :] = calcEuclideanDistance(original, doppelgaenger, val)

            if env == 1:
                atypicalIDX = 0
                typicalIDX = 1
            else:
                atypicalIDX = 1
                typicalIDX = 0
            idx1 = val[atypicalIDX]  # number of specializations towards / deconstructions of atypical env
            idx2 = val[typicalIDX]  # # number of specializations towards / deconstructions of typical env

            atypicalSpec = zip(list(original[:, idx1]), list(doppelgaenger[:, idx1]))
            typicalSpec = zip(original[:, idx2], doppelgaenger[:, idx2])
            atypicalTraces[(key, t)] = atypicalSpec
            typicalTraces[(key, t)] = typicalSpec

    return allResults, cueProbabilities, atypicalTraces, typicalTraces


def postProcessPosteriorBelief(posBeliefOrg, posBeliefDG, cueProbabilities):
    # first calculate difference across separation time; that is the average of the difference matrix
    absDifferencesOrg = np.absolute(np.diff(posBeliefOrg))
    meanDifferenceOrg = np.average(absDifferencesOrg, axis=0, weights=cueProbabilities)
    absDifferencesDG = np.absolute(np.diff(posBeliefDG))
    meanDifferenceDG = np.average(absDifferencesDG, axis=0, weights=cueProbabilities)
    posDifferences = np.abs(posBeliefOrg[:, 1:] - posBeliefDG[:, 1:])
    meanPosDifferences = np.average(posDifferences, axis=0, weights=cueProbabilities)
    return meanDifferenceOrg, meanDifferenceDG, meanPosDifferences


def determineLength(T):
    a = np.arange(0, T + 1, 1)
    b = np.arange(0, T + 1, 1)
    lengthCtr = 0
    for a0 in a:
        for b0 in b:
            if a0 + b0 == T:
                current = multiset_permutations(['0'] * a0 + ['1'] * b0)
                lengthCtr += len(list(current))

    return lengthCtr


def runAdoptionExperiment(T, numAgents, env, pE0, cueReliability, resultsPath, argumentR, argumentP, adoptionType):
    # this function will run twinstudies for a specific parameter combination for each possible moment of adoption

    # absolute phenotypic distance: average distance between numAgents organisms and their doppelgaengers at the end
    # of development

    # proportional distance: absolute distance divided by maximum possible distance
    # maximum possible distance: 20 * sqrt(2)
    tValues = np.arange(1, T + 1, 1)

    resultsBeliefAggr = np.zeros((T, 3))

    posBeliefDiffStart = [0] * T
    posBeliefDiffEnd = [0] * T

    resultLen = determineLength(T)
    if resultLen > numAgents:
        resultLen = int(math.ceil(float(numAgents) / 12)) * 12  # Does this still work?

    results = np.zeros((T, resultLen))
    resultsTempPhenotypes = np.zeros(
        (T, resultLen))  # euclidean distance between original and twin right after exposure

    # new dictionaries for storing the different types of phenotypic distances
    # we are computing phenotypic distances between y0 and y1
    # between y0C and y1C
    # and between y0D and y1D

    allResults = {}
    allResultsTemp = {}
    namedList = {'y0y1': (5, 6), 'y0Cy1C': (0, 1), 'y0Dy1D': (2, 3)}
    for key, val in namedList.items():
        allResults[key] = results.copy()
        allResultsTemp[key] = resultsTempPhenotypes.copy()

    atypicalTraces = {}
    typicalTraces = {}

    for t in tValues:
        print("currently working on time step: " + str(t))
        original, doppelgaenger, posBeliefOrg, posBeliefDG, originalTemp, doppelgaengerTemp, cueProbabilities = runTwinStudiesParallel(
            t,
            numAgents,
            env,
            pE0,
            cueReliability,
            True,
            T,
            resultsPath,
            argumentR,
            argumentP,
            adoptionType)

        if isinstance(cueProbabilities, np.ndarray):
            cueProbabilities = [float(elem) for elem in cueProbabilities]

        if t == 1:
            simNum = posBeliefOrg.shape[0]
            posBeliefOrg[:, 0] = [1 - pE0] * simNum
            posBeliefDG[:, 0] = [1 - pE0] * simNum

        for key, val in namedList.items():
            allResults[key][t - 1, :] = calcEuclideanDistance(original, doppelgaenger, val)
            allResultsTemp[key][t - 1, :] = calcEuclideanDistance(originalTemp, doppelgaengerTemp, val)

            if env == 1:
                atypicalIDX = 0
                typicalIDX = 1
            else:
                atypicalIDX = 1
                typicalIDX = 0
            idx1 = val[atypicalIDX]  # number of specializations towards / deconstructions of atypical env
            idx2 = val[typicalIDX]  # # number of specializations towards / deconstructions of typical env

            atypicalSpec = zip(list(original[:, idx1]), list(doppelgaenger[:, idx1]))
            typicalSpec = zip(original[:, idx2], doppelgaenger[:, idx2])
            atypicalTraces[(key, t)] = atypicalSpec
            typicalTraces[(key, t)] = typicalSpec

        meanDifferenceOrg, meanDifferenceDG, meanPosDifferences = postProcessPosteriorBelief(posBeliefOrg, posBeliefDG,
                                                                                             cueProbabilities)

        if t == 1:
            posBeliefDeltaOrg = meanDifferenceOrg
            posBeliefDeltaOrg = posBeliefDeltaOrg.reshape(T, 1)
            posBeliefDeltaDG = meanDifferenceDG
            posBeliefDeltaDG = posBeliefDeltaDG.reshape(T, 1)

        posBeliefDiffEnd[t - 1] = meanPosDifferences[-1]  # store the last difference
        posBeliefDiffStart[t - 1] = meanPosDifferences[0]  # store the first difference

        # it might still be interesting to have a plot with one line per ontogeny indicating
        # belief change of the orginal in one plot, the doppelgaenger, and the posterior belief change

        # is the absolute average difference across time and agents in posterior belief interesting?
        # I think it might be: it is a different proxy for plasticity in belief
        # how different is twins' belief in environment 1 due to exposure to cues?; focus on this for now, but keep
        # thinking about this
        resultsBeliefAggr[t - 1, :] = [np.mean(meanDifferenceOrg), np.mean(meanDifferenceDG),
                                       np.mean(meanPosDifferences)]

    # need to add the other two columns
    posBeliefDiffEnd = np.array(posBeliefDiffEnd).reshape(T, 1)
    posBeliefDiffStart = np.array(posBeliefDiffStart).reshape(T, 1)

    resultsBeliefAggr = np.hstack(
        (resultsBeliefAggr, posBeliefDeltaOrg, posBeliefDeltaDG, posBeliefDiffEnd, posBeliefDiffStart))

    return allResults, resultsBeliefAggr, allResultsTemp, cueProbabilities, atypicalTraces, typicalTraces


def normRows(vec):
    if vec.min() != vec.max():
        curRowNorm = (vec - vec.min()) / float((vec.max() - vec.min()))
        return curRowNorm
    else:
        return vec


def meanAbsDistance(data, currMean, cueProbabilities):
    dataDiff = data - currMean
    return np.average(abs(dataDiff), weights=cueProbabilities)


def postProcessResultsMat(allResults, T, endOfExposure, lag, cueProbabilities):
    emptyDict = {}
    resultsVecAll = emptyDict.copy()
    resultsVecVarAll = emptyDict.copy()
    resultsVecRelativeAll = emptyDict.copy()
    resultsVecRelativeVarAll = emptyDict.copy()

    for key, results in allResults.items():
        resultsVec = []
        resultsVecVar = []
        resultsVecRelative = []
        resultsVecRelativeVar = []

        if not endOfExposure:  # if phenotypic distance has been measured at the end of ontogeny

            if key == 'y0Dy1D':
                resultsNorm = results / float(int(T / 2) * np.sqrt(2))
            else:
                resultsNorm = results / float(T * np.sqrt(2))
            for idx in range(results.shape[0]):
                curRowNorm = resultsNorm[idx, :]
                curRow = results[idx, :]
                if key == 'y0Dy1D':
                    curRowRelative = curRow / float(int(((T - idx)) / 2) * np.sqrt(2))
                else:
                    curRowRelative = curRow / float((T - idx) * np.sqrt(2))

                resultsVec.append(
                    np.average(curRowNorm, weights=cueProbabilities))  # average euclidean distance across agents
                resultsVecRelative.append(np.average(curRowRelative, weights=cueProbabilities))
                varRel = meanAbsDistance(curRowRelative, resultsVecRelative[-1], cueProbabilities)
                varAbs = meanAbsDistance(curRowNorm, resultsVec[-1], cueProbabilities)
                resultsVecVar.append(varAbs)
                resultsVecRelativeVar.append(varRel)
        else:
            for idx in range(results.shape[0]):
                curRow = results[idx, :]
                if key == 'y0Dy1D':  # need to think about rounding
                    curRowNorm = curRow / float(int((lag + idx) / 2) * np.sqrt(2))
                    curRowRelative = curRow / float(int(lag / 2) * np.sqrt(2))
                else:
                    curRowNorm = curRow / float((lag + idx) * np.sqrt(2))
                    curRowRelative = curRow / float(lag * np.sqrt(2))
                resultsVec.append(np.average(curRowNorm, weights=cueProbabilities))
                resultsVecRelative.append(np.average(curRowRelative, weights=cueProbabilities))
                varRel = meanAbsDistance(curRowRelative, resultsVecRelative[-1], cueProbabilities)
                varAbs = meanAbsDistance(curRowNorm, resultsVec[-1], cueProbabilities)
                resultsVecVar.append(varAbs)
                resultsVecRelativeVar.append(varRel)
        resultsVecAll[key] = resultsVec
        resultsVecVarAll[key] = resultsVecVar
        resultsVecRelativeAll[key] = resultsVecRelative
        resultsVecRelativeVarAll[key] = resultsVecRelativeVar

    return resultsVecAll, resultsVecRelativeAll, resultsVecVarAll, resultsVecRelativeVarAll


def rescaleNumbers(newMin, newMax, numbersArray):
    OldMin = np.min(numbersArray)
    OldMax = np.max(numbersArray)
    result = [(((OldValue - OldMin) * (newMax - newMin)) / float(OldMax - OldMin)) + newMin for OldValue in
              numbersArray]
    return result


def area_calc(probs, r):
    # result = [(p)**2 * np.pi*r for p in probs]
    result = [np.sqrt(float(p)) * r for p in probs]
    return result


def calcPointToPlot(focalPoint, idx):
    sidePoints = (10 - focalPoint) / float(3)
    focalResult = 10 - (2 * sidePoints)
    result = [sidePoints] * 3
    result[idx] = focalResult
    return result


def plotFitnessDifferenceOverviewMerge(cueValidityArr, T, adultTArr, autoCorrDict,
                                       twinResultsAggregatedPath, mainDataPath, argumentR, argumentP,
                                       levelsAutoCorrToPlot, nameArg):
    if len(nameArg) == 3:
        nameArg = nameArg[0:2]

    rowVec = []
    for currX in nameArg:
        for currY in adultTArr:
            rowVec.append((currX, currY))

    # prepare the x-axis values
    fig, axes = plt.subplots(len(cueValidityArr), len(rowVec), sharex=True, sharey=True)
    fig.set_size_inches(32, 16)
    ax_list = fig.axes

    jx = 0
    for symmArg, adultT in rowVec:  # one column per adultT
        ix = 0
        for cueVal in cueValidityArr:  # one row per cue reliability value
            ax = ax_list[ix * len(rowVec) + jx]
            ax.set(aspect=4)
            plt.sca(ax)

            autoCorrDict_sorted = sorted(autoCorrDict[symmArg].items(), key=operator.itemgetter(1))
            autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
            dataPath = os.path.join(mainDataPath, str(symmArg))

            dataPath0 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
            dataPath1 = os.path.join(dataPath,
                                     '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))

            # first load the data
            fileName = "fitnessDifferences.p"
            filePath0 = os.path.join(dataPath0, fileName)
            filePath1 = os.path.join(dataPath1, fileName)

            if os.path.exists(filePath0):  # contains mature phenotypes and cueProb for specific comb. of mc and cue rel
                fitnessDifferences = pickle.load(open(filePath0, 'rb'))
            if os.path.exists(filePath1):
                fitnessDifferences.update(pickle.load(open(filePath1, 'rb')))

            # for the current cueVal load the distancedictionaries
            cueValDict = {val: fitnessDifferences[(key, cueVal)] for key, val in autoCorrDict_sorted}

            if levelsAutoCorrToPlot:
                # the next line find the indices of the closest autocorrelation values that match the user input
                idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
                       levelsAutoCorrToPlot]
                autoCorrValSubset = np.array(autoCorrVal)[idx]
                fitnessDifferences = []
                for idx, autoCorr in enumerate(autoCorrValSubset):
                    fitnessDifferences += list(cueValDict[autoCorr])
                x = np.arange(len(fitnessDifferences))

                xLabels = ["|S", "O\n\n%s" % round(autoCorrValSubset[0], 1), "G|", "|S",
                           "O\n\n%s" % round(autoCorrValSubset[1], 1), "G|",
                           "|S",
                           "O\n\n%s" % round(autoCorrValSubset[2], 1), "G|"]

                # splot for symmetyric and asymmetric
                barList = plt.bar(x, fitnessDifferences)

                lightGreys = np.arange(0, len(x) - 1, 3)
                greys = np.arange(1, len(x) - 1, 3)
                for colorIdx in range(len(x)):
                    if colorIdx in lightGreys:
                        barList[colorIdx].set_color("lightgray")
                    elif colorIdx in greys:
                        barList[colorIdx].set_color("grey")
                    else:
                        barList[colorIdx].set_color("black")

            else:
                """
                in case that the user did not specify values to pick, compute an average
                - first need to calculate cutoff points
                """
                extremeIDX = np.floor(len(autoCorrVal) / float(3))
                midIDX = np.ceil(len(autoCorrVal) / float(3))
                loopArrayIDX = [extremeIDX, midIDX, extremeIDX]
                loopArrayLevl = ['low', 'moderate', 'high']

                cueValDictSubset = {}
                for idx in range(len(loopArrayIDX)):
                    levl = loopArrayLevl[idx]

                    if idx == 0:
                        startIdx = int(idx)
                    else:
                        startIdx = int(endIdx)
                    endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                    autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]
                    fitnessDifferenceMean = np.mean([cueValDict[autoCorr] for autoCorr in autoCorrValSubset], axis=0)
                    cueValDictSubset[levl] = fitnessDifferenceMean

                fitnessDifferences = []
                for idx, autoCorr in enumerate(loopArrayLevl):
                    fitnessDifferences += list(cueValDictSubset[autoCorr])
                x = np.arange(len(fitnessDifferences))

                xLabels = ["|S", "O\n\n%s" % loopArrayLevl[0][0].upper(), "G|", "|S",
                           "O\n\n%s" % loopArrayLevl[1][0].upper(), "G|",
                           "|S",
                           "O\n\n%s" % loopArrayLevl[2][0].upper(), "G|"]

                barList = plt.bar(x, fitnessDifferences)

                lightGreys = np.arange(0, len(x) - 1, 3)
                greys = np.arange(1, len(x) - 1, 3)
                for colorIdx in range(len(x)):
                    if colorIdx in lightGreys:
                        barList[colorIdx].set_color("lightgray")
                    elif colorIdx in greys:
                        barList[colorIdx].set_color("grey")
                    else:
                        barList[colorIdx].set_color("black")
            """
            plot two parallels to the x-axis to highlight the 0 and 1 mark
            """
            xLines = np.arange(-0.5, 8.6, 0.1)

            plt.plot(xLines, [1] * len(xLines), linestyle='dashed', linewidth=1, color='grey')
            plt.plot(xLines, [-1] * len(xLines), linestyle='dashed', linewidth=1, color='grey')

            yvals = np.arange(-0.9, 1, 0.1)
            plt.plot([2.5] * len(yvals), yvals, linestyle='dashed', linewidth=1, color='grey')
            plt.plot([5.5] * len(yvals), yvals, linestyle='dashed', linewidth=1, color='grey')

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-1, 1)

            if (jx + 1) % len(adultTArr) == 0 and not jx == (len(rowVec)) - 1:
                paramVLine = max(x) + 1.5
                ax.vlines([paramVLine], -0.05, 1.05, transform=ax.get_xaxis_transform(), color='black', lw=2,
                          clip_on=False)

            if ix == 0:
                plt.title("%s     " % str(adultT), fontsize=25, pad=15, loc='center')

            if ix == len(cueValidityArr) - 1:
                plt.xlabel('', fontsize=25, labelpad=10)
                plt.tick_params(pad=10)

                plt.xticks(x, xLabels, fontsize=20)

            else:
                ax.get_xaxis().set_visible(False)

            if jx == 0:
                plt.yticks([-1, 0, 1], fontsize=20)
                if ix == len(cueValidityArr) - 1:
                    plt.ylabel('fitness difference', fontsize=25, labelpad=20)
            else:
                ax.tick_params(axis='y', which='both', length=0)

            if jx == len(rowVec) - 1:
                plt.ylabel(str(cueVal), labelpad=15, rotation='vertical', fontsize=25)
                ax.yaxis.set_label_position("right")

            ix += 1
        jx += 1

    fig.text(0.14, 0.01, 'autocorrelation', fontsize=25, horizontalalignment='left', verticalalignment='bottom',
             transform=ax.transAxes, rotation='horizontal')

    top = 0.8
    fig.text(0.94, 0.45, 'cue reliability', fontsize=25, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    autoCorrCoord = 0.875

    plt.subplots_adjust(wspace=0.0, hspace=0.0, bottom=0.1, top=top)

    fig.text(0.514, 0.95, 'transition probabilities', fontsize=25, horizontalalignment='center',
             verticalalignment='center', transform=ax.transAxes, rotation='horizontal')

    figVal = 1 / float((len(nameArg)))
    halfFigVal = figVal / float(2)
    figVals = np.arange(halfFigVal, 1, figVal)
    nameArg2 = ["symmetric", "asymmetric"]

    for figCoord, adultT in zip(figVals, nameArg2):
        if len(figVals) == 2:
            if figCoord < 0.5:
                figCoordF = figCoord + 0.064
            else:
                figCoordF = figCoord - 0.05
        else:
            if figCoord < 0.3:
                figCoordF = figCoord + 0.085
            elif figCoord > 0.3 and figCoord < 0.6:
                figCoordF = 0.514
            else:
                figCoordF = figCoord - 0.055
        fig.text(figCoordF, autoCorrCoord, '%s\n\n\nadult life span' % adultT, fontsize=25,
                 horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes, rotation='horizontal')

    plt.savefig(os.path.join(twinResultsAggregatedPath, 'fitnessDifferencesOverviewMerge%s.pdf' % levelsAutoCorrToPlot),
                bbox_inches='tight', dpi=600)
    plt.close()


def plotPlasticityCurvesOverview33(cueValidityArr, T, adultTArr, env, autoCorrDict,
                                   twinResultsAggregatedPath, mainDataPath, argumentR, argumentP, adoptionType, lag,
                                   endOfExposure,
                                   studyArg, priorArr, nameArg):
    T = T - 1
    arg = "relative"  # choose whether you want this plot for relative or absolute distance

    linestyle_tuple = [
        ('solid', (0, ())),
        ('dotted', (0, (1, 1))),
        ('densely dashed', (0, (5, 1))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),
        ('loosely dotted', (0, (1, 10))),
        ('loosely dashed', (0, (5, 10))),
        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('densely dotted', (0, (1, 1))),
        ('dashed', (0, (5, 5))),
        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)))]

    levelsAutoCorrToPlot = [0.2, 0.5, 0.8]  # TODO change this back to None

    # prepare the x-axis values
    tValues = np.arange(1, T + 1, 1)
    fig, axes = plt.subplots(len(cueValidityArr), len(priorArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    jx = 0
    for prior in priorArr:  # one plot for relative, one plot for absolute phenotypic distance
        if prior == '05':
            levelsAutoCorrToPlot = [0.2, 0.5, 0.77]  # otherwise it picks 0.85 instead of 0.75
        else:
            levelsAutoCorrToPlot = [0.2, 0.5, 0.8]

        if 'E' in prior:
            env = prior[-2]

        # here select the right autocorr dict and dataPath
        autoCorrDict_sorted = sorted(autoCorrDict[prior].items(), key=operator.itemgetter(1))
        autoCorrKeys, autoCorrVal = zip(*autoCorrDict_sorted)
        dataPath = os.path.join(mainDataPath, str(prior))

        ix = 0
        for cueVal in cueValidityArr:
            ax = ax_list[ix * len(priorArr) + jx]

            plt.sca(ax)
            for i, adultT in enumerate(adultTArr):  # one line per adult T
                # linestyle depends on current adult T value
                linestyle = linestyle_tuple[i][1]
                dataPath0 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half0_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))
                dataPath1 = os.path.join(dataPath,
                                         '%s/PlottingResults_Half1_%s_%sNew' % (adultT, argumentR[0], argumentP[0]))

                # first load the data
                fileName = '%sDistanceDict%s%s%s%s_%s.p' % (
                    arg, studyArg, adoptionType, lag, endOfExposure, env)
                filePath0 = os.path.join(dataPath0, fileName)
                filePath1 = os.path.join(dataPath1, fileName)

                if os.path.exists(filePath0):
                    distanceDict = pickle.load(open(filePath0, 'rb'))
                    if os.path.exists(filePath1):
                        distanceDict.update(pickle.load(open(filePath1, 'rb')))
                else:
                    print('no data available')
                    exit()

                # for he current cueVal load the distancedictionaries
                cueValDict = {val: distanceDict[(key, cueVal)] for key, val in autoCorrDict_sorted}

                if levelsAutoCorrToPlot:
                    # the next line find the indices of the closest autocorrelation values that match the user input
                    idx = [min(range(len(autoCorrVal)), key=lambda i: abs(autoCorrVal[i] - x)) for x in
                           levelsAutoCorrToPlot]
                    autoCorrValSubset = np.array(autoCorrVal)[idx]
                    [plt.plot(tValues, cueValDict[autoCorr], linestyle=linestyle, linewidth=2, markersize=5,
                              marker='o', color=str(1 - autoCorr - 0.1), markerfacecolor=str(1 - autoCorr - 0.1),
                              label="%s & %s" % (autoCorr, adultT)) for autoCorr in autoCorrValSubset]

                else:
                    """
                    in case that the user did not specify values to pick, compute an average
                    - first need to calculate cutoff points
                    """
                    extremeIDX = np.floor(len(autoCorrVal) / float(3))
                    midIDX = np.ceil(len(autoCorrVal) / float(3))
                    loopArrayIDX = [extremeIDX, midIDX, extremeIDX]
                    loopArrayLevl = ['low', 'moderate', 'high']

                    cueValDictSubset = {}
                    for idx in range(len(loopArrayIDX)):
                        levl = loopArrayLevl[idx]

                        if idx == 0:
                            startIdx = int(idx)
                        else:
                            startIdx = int(endIdx)
                        endIdx = int(sum(loopArrayIDX[0:idx + 1]))

                        autoCorrValSubset = np.array(autoCorrVal)[startIdx:endIdx]
                        plastcityVal = np.mean([cueValDict[autoCorr] for autoCorr in autoCorrValSubset], axis=0)
                        cueValDictSubset[levl] = plastcityVal

                    colorArr = [0.1, 0.5, 0.9]
                    [plt.plot(tValues, cueValDictSubset[autoCorr], linestyle=linestyle, linewidth=2, markersize=5,
                              marker='o', color=str(1 - colorArr[idx] - 0.1),
                              markerfacecolor=str(1 - colorArr[idx] - 0.1),
                              label="%s & %s" % (autoCorr[0].upper(), adultT)) for idx, autoCorr in
                     enumerate(loopArrayLevl)]

            """
            plot two parallels to the x-axis to highlight the 0 and 1 mark
            """
            plt.plot(tValues, [0] * T, linestyle='dashed', linewidth=1, color='grey')
            plt.plot(tValues, [1] * T, linestyle='dashed', linewidth=1, color='grey')

            if ix == len(cueValidityArr) - 1 and jx == 0:
                anchPar = len(priorArr) / float(2)
                legend = ax.legend(loc='upper center', bbox_to_anchor=(anchPar, -0.3),
                                   title='autocorrelation & adult lifespan',
                                   ncol=len(adultTArr), fancybox=True, shadow=False, fontsize=20)
                plt.setp(legend.get_title(), fontsize='20')

            plt.suptitle(nameArg, fontsize=20)
            if ix == 0:
                plt.title("%s" % (prior), fontsize=20)

            # stuff to amke the plot look pretty
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-0.05, 1.05)
            plt.yticks(np.arange(0, 1.1, 0.2), fontsize=15)
            plt.xticks([])

            if ix == len(cueValidityArr) - 1 and jx == 0:
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)

                plt.xlabel('ontogeny', fontsize=20, labelpad=20)

            if jx == len(priorArr) - 1:
                ax.yaxis.set_label_position("right")
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)

            ix += 1
        jx += 1
    plt.subplots_adjust(wspace=0.05, hspace=0.1, bottom=0.2, top=0.90)
    fig.text(0.98, 0.54, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
             transform=ax.transAxes, rotation='vertical')
    fig.text(0.08, 0.31, 'phenotypic distance', fontsize=20, ha='center', va='center', rotation='vertical')
    if endOfExposure:
        safeStr = "EndOfExposure"
    else:
        safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsAggregatedPath,
                     '%s_%s_%s_%s_%s%sPlasticityOverviewTotal33%s.jpg' % (
                         studyArg, adoptionType, lag, safeStr, env, False, levelsAutoCorrToPlot)),
        dpi=800)

    plt.close()





def plotTriangularPlotsArrow(tValues, priorE0Arr, cueValidityArr, maturePhenotypes, T, twinResultsPath, env):
    # for ietrating through the different phenotypic dimensions

    namedList = {'y0y1': (5, 6), 'y0Cy1C': (0, 1),'y0Dy1D': (2, 3)} #
    namedListColors = {'y0y1': ('black', 0.8, 300), 'y0Cy1C': ('blue', 0.4, 300),'y0Dy1D': ('green', 0.4, 300)} #
    #250,400,600
    fig, axes = plt.subplots(len(cueValidityArr), len(priorE0Arr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes
    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]
            ax.set(aspect='equal')  # if you remove this, plots won't be square
            plt.sca(ax)

            """
            Here goes the actual plotting code 
            """
            maturePhenotypesCurrAll, cueProbabilities = maturePhenotypes[(pE0, cueVal)]


            for key, val in namedList.items():
                color = namedListColors[key][0]
                alpha = namedListColors[key][1]
                scaleFactor = namedListColors[key][2]
                maturePhenotypesCurr = np.transpose(
                    [maturePhenotypesCurrAll[:, val[0]], maturePhenotypesCurrAll[:, val[1]],
                     maturePhenotypesCurrAll[:, 4]])
                numAgents = maturePhenotypesCurr.shape[0]
                # now need to work on the scaling of points

                unique, uniqueCounts = np.unique(maturePhenotypesCurr[:,[0,1,2]], axis=0, return_counts=True)


                if isinstance(cueProbabilities, list) or isinstance(cueProbabilities, np.ndarray):
                    uniqueFrac = []
                    for matPhen in unique:
                        probIdx = np.where(
                            (maturePhenotypesCurr[:, 0] == matPhen[0]) & (maturePhenotypesCurr[:, 1] == matPhen[1]) & (
                                    maturePhenotypesCurr[:, 2] == matPhen[2]))[0]
                        uniqueFrac.append(sum(cueProbabilities[probIdx]))

                    uniqueFrac = [float(elem) for elem in uniqueFrac]
                    area2 = np.array(uniqueFrac) * float(scaleFactor)


                else:
                    uniqueFrac = uniqueCounts / float(numAgents)
                    area2 = uniqueFrac* scaleFactor

                if key == 'y0Cy1C':
                    constructionPhen = unique
                    constructionArea = area2


                elif key == 'y0y1':
                    totalPhen = unique
                    totalArea = area2
                    #adding the waiting
                    ax.scatter(20-unique[:, 1], 20-unique[:, 0], s=30, color=color, alpha=(unique[:,2])/20,
                               edgecolors='none', marker = 's')

            uniqueCompletePhen,counts = np.unique(maturePhenotypesCurrAll, axis=0, return_counts=True)

            noDPhen = uniqueCompletePhen[np.where((uniqueCompletePhen[:, 2] == 0) * (uniqueCompletePhen[:, 3] == 0))]

            noDIDX = [idx for idx in range(len(uniqueCompletePhen)) if
                      uniqueCompletePhen[idx, :].tolist() in noDPhen.tolist()]

            percentD = round((1 - (sum(counts[noDIDX]) / float(numAgents))) * 100, 2)

            noDPhen = noDPhen[:, [0, 1, 4]]

            # plotting construction and total phenotypes
            # starting with construction
            noDIDX = [idx for idx in range(len(constructionPhen)) if
                      constructionPhen[idx, :].tolist() in noDPhen.tolist()]
            DIDX = [idx for idx in range(len(constructionPhen)) if
                    constructionPhen[idx, :].tolist() not in noDPhen.tolist()]

            unique = constructionPhen[noDIDX, :]
            area2 = constructionArea[noDIDX]
            ax.scatter(unique[:, 0], unique[:, 1], s=area2 * 1.4, color='lightgrey', alpha=.6,
                       edgecolors='black', zorder=2)

            unique = constructionPhen[DIDX, :]
            area2 = constructionArea[DIDX]
            ax.scatter(unique[:, 0], unique[:, 1], s=area2 * 1.4, color='teal', alpha=.7,
                       edgecolors='none', zorder=3)

            # next plot the total phenotypes
            DIDX = [idx for idx in range(len(totalPhen)) if
                    totalPhen[idx, :].tolist() not in noDPhen.tolist()]
            unique = totalPhen[DIDX, :]
            area2 = totalArea[DIDX]
            ax.scatter(unique[:, 0], unique[:, 1], s=area2 * 1.4, color='indianred', alpha=.7,
                       edgecolors='none', zorder=4)

            #draw arrows
            uniqueCompletePhenX = uniqueCompletePhen[:,[5,6]]
            uniqueCompletePhenY = uniqueCompletePhen[:, [0, 1]]

            [ax.annotate('',[uniqueCompletePhenX[idx][0], uniqueCompletePhenX[idx][1]],
                                   [uniqueCompletePhenY[idx][0], uniqueCompletePhenY[idx][1]], arrowprops = {'arrowstyle':"->",
                                   'color':"black", 'lw':0.15,'alpha' : 0.5, 'mutation_scale':5,'connectionstyle':'Arc3','shrinkA':3, 'shrinkB':3, 'zorder': 1}) for idx in range(len(uniqueCompletePhen))]


            # add diagonal
            #ax.plot(range(0,21), range(20,-1,-1), color = 'k', lw = 1, zorder = -1)

            ax.axes.set_xlim(left=-1, right=max(tValues)+1)
            ax.axes.set_ylim(bottom=-1, top=max(tValues)+1)

            if ix == 0:
                plt.title("%s\n\ndeconstruction: %s%%" % ((1 - pE0), percentD), fontsize=20)
            else:
                plt.title("deconstruction: %s%%" % percentD, fontsize=20)


            if jx == len(priorE0Arr) - 1:
                ax.set_ylabel(str(cueVal), labelpad=10, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            labelVals = np.arange(0, max(tValues) + 1, 2)
            # set the ticks
            labels = []
            for i in range(len(labelVals)):
                if i % 2 == 0:
                    labels.append(labelVals[i])
                else:
                    labels.append("")



            if jx == 0 and ix == len(cueValidityArr)-1:
                fontsize = 20
                ax.set_xlabel('y0', fontsize=fontsize, labelpad=10)
                ax.set_ylabel('y1', fontsize=fontsize, labelpad=10)

            ax.set_xticks(np.arange(0, max(tValues) + 1, 2), labels, fontsize=15, rotation=0)
            ax.set_yticks(np.arange(0, max(tValues) + 1, 2), labels, fontsize=15, rotation=0)


            jx += 1
        ix += 1
        plt.suptitle('prior estimate', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 rotation='vertical')

    #plt.savefig(os.path.join(twinResultsPath, 'ternaryArrow_%s.pdf' % env), dpi=600)
    plt.savefig(os.path.join(twinResultsPath, 'ternaryArrow_%s.png' % env), dpi=600)

    plt.close()

def plotTriangularPlots(tValues, priorE0Arr, cueValidityArr, maturePhenotypes, T, twinResultsPath, env):
    # for ietrating through the different phenotypic dimensions

    namedList = {'y0y1': (5, 6), 'y0Cy1C': (0, 1), 'y0Dy1D': (2, 3)}
    namedListColors = {'y0y1': ('black', 0.6, 300), 'y0Cy1C': ('blue', 0.4, 500), 'y0Dy1D': ('green', 0.4, 500)}
    #250,400,600
    fig, axes = plt.subplots(len(cueValidityArr), len(priorE0Arr), sharex=True, sharey=True,
                             subplot_kw={"projection": "3d", 'computed_zorder': False})
    fig.set_size_inches(16, 16)
    ax_list = fig.axes
    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]
            ax.set(aspect='equal')  # if you remove this, plots won't be square
            plt.sca(ax)

            """
            Here goes the actual plotting code 
            """
            maturePhenotypesCurrAll, cueProbabilities = maturePhenotypes[(pE0, cueVal)]


            zorder = 0
            for key, val in namedList.items():
                color = namedListColors[key][0]
                alpha = namedListColors[key][1]
                scaleFactor = namedListColors[key][2]
                maturePhenotypesCurr = np.transpose(
                    [maturePhenotypesCurrAll[:, val[0]], maturePhenotypesCurrAll[:, val[1]],
                     maturePhenotypesCurrAll[:, 4]])
                numAgents = maturePhenotypesCurr.shape[0]
                # now need to work on the scaling of points

                unique, uniqueCounts = np.unique(maturePhenotypesCurr, axis=0, return_counts=True)

                if isinstance(cueProbabilities, list) or isinstance(cueProbabilities, np.ndarray):
                    uniqueFrac = []
                    for matPhen in unique:
                        probIdx = np.where(
                            (maturePhenotypesCurr[:, 0] == matPhen[0]) & (maturePhenotypesCurr[:, 1] == matPhen[1]) & (
                                    maturePhenotypesCurr[:, 2] == matPhen[2]))[0]
                        uniqueFrac.append(sum(cueProbabilities[probIdx]))

                    uniqueFrac = [float(elem) for elem in uniqueFrac]
                    area2 = np.array(uniqueFrac) * float(scaleFactor)


                else:
                    uniqueFrac = uniqueCounts / float(numAgents)
                    area2 = uniqueFrac* scaleFactor

                # this one would be scalling according to area

                ax.scatter3D(unique[:, 0], unique[:, 1], unique[:, 2], s=area2*1.3, color=color, alpha=alpha,
                           edgecolors='none', zorder = zorder)
                zorder +=1

            ax.axes.set_xlim3d(left=0, right=max(tValues))
            ax.axes.set_ylim3d(bottom=0, top=max(tValues))
            ax.axes.set_zlim3d(bottom=0, top=11)

            if ix == 0:
                plt.title("%s" % (1 - pE0), fontsize=20)

            if jx == len(priorE0Arr) - 1:
                ax.set_zlabel(str(cueVal), labelpad=10, rotation='vertical', fontsize=20)

            labelVals = np.arange(0, max(tValues) + 1, 2)
            # set the ticks
            labels = []
            for i in range(len(labelVals)):
                if i % 2 == 0:
                    labels.append(labelVals[i])
                else:
                    labels.append("")

            labelValsZ = np.arange(0,11, 2)
            # set the ticks
            labelsZ = []
            for i in labelValsZ:
                if i % 2 == 0 and i != 0:
                    labelsZ.append(i)
                else:
                    labelsZ.append("")

            jx += 1
        ix += 1
        plt.suptitle('prior estimate', fontsize=20)

        #labeling of the subplots
        fontsize = 20
        positions = ['none', 'none', 'none', 'none', 'none','none', 'default', 'none', 'none']
        counter = 0
        for ax, pos in zip(axes.flatten(), positions):
            for axis in ax.xaxis, ax.yaxis, ax.zaxis:
                axis.set_label_position(pos)
                axis.set_ticks_position(pos)
            if counter == 6:
                ax.set_xlabel('y0', fontsize=fontsize, labelpad=10)
                ax.set_ylabel('y1', fontsize=fontsize, labelpad=10)
                ax.set_zlabel('waiting', fontsize=fontsize, rotation='vertical', labelpad=5)

                ax.set_zticks(np.arange(0, 11, 2), labelsZ, fontsize=13, rotation=0)
                ax.set_xticks(np.arange(0, max(tValues) + 1, 2), labels, fontsize=13, rotation=0)
                ax.set_yticks(np.arange(0, max(tValues) + 1, 2), labels, fontsize=13, rotation=0,
                              verticalalignment='baseline',
                              horizontalalignment='left')
            ax.set(aspect='equal')
            counter += 1

        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 rotation='vertical')

    plt.savefig(os.path.join(twinResultsPath, 'ternary_%s.pdf' % env), dpi=600)
    plt.close()


def fitnessFunc(state, b0_D, b1_D, argumentR, argumentP, T, beta, psi_weighting):
    y0, y1, yw = state

    if argumentR == 'linear':
        phiVar = b0_D * D(str(y0)) + b1_D * D(str(y1))

    elif argumentR == 'diminishing':
        alphaRD = D(str(T)) / (D(str(1)) - (np.exp(D(str(-beta)) * (D(str(T))))))
        phiVar = b0_D * alphaRD * (D(str(1)) - np.exp(-D(str(beta)) * D(str(y0)))) + b1_D * alphaRD * (D(str(1)) -
                                                                                                       np.exp(-D(str(
                                                                                                           beta)) * D(
                                                                                                           str(y1))))

    elif argumentR == 'increasing':
        alphaRI = D(str(T)) / ((np.exp(D(str(beta)) * D(str(T)))) - D(str(1)))
        phiVar = b0_D * alphaRI * (np.exp(D(str(beta)) * D(str(y0))) - D(str(1))) + b1_D * alphaRI * (
                np.exp(D(str(beta)) * D(str(y1))) - D(str(1)))
    else:
        print('Wrong input argument to additive fitness reward function')
        print('Argument must be linear, increasing or diminishing')
        exit(1)

    if argumentP == 'linear':
        psiVar = -(b0_D * D(str(y1)) + b1_D * D(str(y0)))
    elif argumentP == 'diminishing':
        alphaPD = D(str(T)) / (D(str(1)) - (np.exp(D(str(-beta)) * (D(str(T))))))
        psiVar = -(b0_D * alphaPD * (D(str(1)) - np.exp(-D(str(beta)) * D(str(y1)))) + b1_D * alphaPD * (D(str(1)) -
                                                                                                         np.exp(-D(str(
                                                                                                             beta)) * D(
                                                                                                             str(y0)))))


    elif argumentP == 'increasing':

        alphaPI = D(str(T)) / ((np.exp(D(str(beta)) * D(str(T)))) - D(str(1)))
        psiVar = -(b0_D * alphaPI * (np.exp(D(str(beta)) * D(str(y1))) - D(str(1))) + b1_D * alphaPI * (
                np.exp(D(str(beta)) * D(str(y0))) - D(str(1))))


    else:
        print('Wrong input argument to additive fitness penalty function')
        print('Argument must be linear, increasing or diminishing')
        exit(1)

    tf = D(str(0)) + phiVar + D(str(psi_weighting)) * psiVar

    return tf


def calcFitness(state, argumentR, argumentP, posE1, T, beta, psi_weighting):
    b1_D = D(str(posE1))
    b0_D = D(str(1)) - D(str(b1_D))

    tfCurr = fitnessFunc(state, b0_D, b1_D, argumentR, argumentP, T, beta, psi_weighting)

    return tfCurr


def fitnessDifference(priorE0Arr, cueValidityArr, policyPath, T, resultsPath, baselineFitness,
                      argumentR,
                      argumentP, numAgents):
    # fitness functions
    # keep in mind fitnessMax is equivalent to T

    beta = 0.2
    # dictionary for storing the results
    resultsDict = {}
    for pE0 in priorE0Arr:
        for cueReliability in cueValidityArr:

            print("Currently calculating expected fitness differences with pE0: " + str(
                pE0) + " and cue reliability: " + str(cueReliability))

            # load the optimal policy
            policyPathData = os.path.join(policyPath,
                                          'runTest_%s%s_%s%s/finalRaw.csv' % (
                                              argumentR[0], argumentP[0], pE0, cueReliability))
            policy = pd.read_csv(policyPathData, index_col=0).reset_index(drop=True)

            subDf = policy[policy['time'] == T].reset_index(drop=True)

            tf = subDf['fitness'].values
            tf = [float(val) for val in tf]
            posE1 = subDf['pE1'].values
            posE1 = [float(val) for val in posE1]
            stateProb = subDf['stateProb'].values
            stateProb = [float(val) for val in stateProb]

            # fitness following the optimal policy
            OFitness = D(str(np.average(tf, weights=stateProb))) / D(str((T)))

            # next specialist Fitness
            if pE0 < 0.5:
                specialistPhenotypes = np.array([0, T, 0])
                SFitnessArrEnv1 = [calcFitness(specialistPhenotypes, argumentR, argumentP, posE1Curr, T, beta, 1)
                                   for posE1Curr in [1] * len(posE1)]

                SFitnessArrEnv1 = [float(val) for val in SFitnessArrEnv1]

                SFitnessArrEnv0 = [calcFitness(specialistPhenotypes, argumentR, argumentP, posE1Curr, T, beta, 1)
                                   for posE1Curr in [0] * len(posE1)]
                SFitnessArrEnv0 = [float(val) for val in SFitnessArrEnv0]

                SFitness = (pE0 * np.average(SFitnessArrEnv0) + (1 - pE0) * np.average(SFitnessArrEnv1)) / T


            else:

                resultLen = len(stateProb)

                specialistPhenotypes = np.zeros((resultLen, 3))

                specialistPhenotypes[:, 0] = np.append(np.array([T] * int(resultLen / 2)),
                                                       np.array([0] * (resultLen - int(resultLen / 2))))

                specialistPhenotypes[:, 1] = np.append(np.array([0] * int(resultLen / 2)),
                                                       np.array([T] * (resultLen - int(resultLen / 2))))

                SFitnessArrEnv1 = [calcFitness(phenotypeS, argumentR, argumentP, posE1Curr, T, beta, 1) for
                                   phenotypeS, posE1Curr in zip(specialistPhenotypes, [1] * len(posE1))]

                SFitnessArrEnv1 = [float(val) for val in SFitnessArrEnv1]

                SFitnessArrEnv0 = [calcFitness(phenotypeS, argumentR, argumentP, posE1Curr, T, beta, 1) for
                                   phenotypeS, posE1Curr in zip(specialistPhenotypes, [0] * len(posE1))]

                SFitnessArrEnv0 = [float(val) for val in SFitnessArrEnv0]
                SFitness = (pE0 * np.average(SFitnessArrEnv0) + (1 - pE0) * np.average(SFitnessArrEnv1)) / T

            phenotypeG = np.array([T / float(2), T / float(2), 0])

            GFitnessArrEnv1 = [
                calcFitness(phenotypeG, argumentR, argumentP, posE1Curr, T, beta, 1)
                for posE1Curr in [1] * len(posE1)]

            GFitnessArrEnv1 = [float(val) for val in GFitnessArrEnv1]

            GFitnessArrEnv0 = [
                calcFitness(phenotypeG, argumentR, argumentP, posE1Curr, T, beta, 1)
                for posE1Curr in [0] * len(posE1)]

            GFitnessArrEnv0 = [float(val) for val in GFitnessArrEnv0]

            GFitness = (pE0 * np.average(GFitnessArrEnv0) + (1 - pE0) * np.average(GFitnessArrEnv1)) / T

            resultsDict[(pE0, cueReliability)] = np.array([SFitness, OFitness, GFitness])

    pickle.dump(resultsDict, open(os.path.join(resultsPath, "fitnessDifferences.p"), "wb"))
    del resultsDict


def plotFitnessDifference(priorE0Arr, cueValidityArr, twinResultsPath):
    # first open the dictionary containing the results

    differencesDict = pickle.load(open(os.path.join(twinResultsPath, "fitnessDifferences.p"), "rb"))
    # define the xAxis
    x = np.arange(3)
    xLabels = ["S", "O", "G"]
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
            fitnessDifferences = differencesDict[(pE0, cueVal)]

            barList = plt.bar(x, fitnessDifferences)

            barList[0].set_color("lightgray")
            barList[1].set_color("grey")
            barList[2].set_color("black")

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
                plt.xticks(x, xLabels, fontsize=15)

            else:
                ax.get_xaxis().set_visible(False)
            if jx == 0:
                plt.ylabel('Fitness difference', fontsize=20, labelpad=10)
                plt.yticks([-1, 0, 1], fontsize=15)

            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=15, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            jx += 1
        ix += 1

        plt.suptitle('markov probabilities', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 rotation='vertical')
    plt.savefig(os.path.join(twinResultsPath, 'fitnessDifferences.png'), dpi=600)
    plt.close()


def calcNegativeRankSwitches(rankDf, T, arg, probabilities):
    tValues = np.arange(0, T, 1)
    results = np.zeros((T, T))
    rankDfNorm = rankDf / (tValues + 2)
    # need possible number of ranks at each time step
    for t in tValues:
        rankDfDiff = rankDf.loc[:, t:].sub(rankDf.loc[:, t], axis=0)

        rankDfDiff2 = rankDfDiff.copy(deep=True)
        if arg == 'unstable':
            rankDfDiff[rankDfDiff2 == 0] = 0
            rankDfDiff[rankDfDiff2 != 0] = 1
        else:
            rankDfDiff[rankDfDiff2 != 0] = 0
            rankDfDiff[rankDfDiff2 == 0] = 1
        results[t, t:] = np.average(rankDfDiff, axis=0, weights=probabilities)

    return results


def plotRankOrderStability(priorE0Arr, cueValidityArr, twinResultsPath, T, types):
    for distFun in types:
        plotRankOrderStability2(priorE0Arr, cueValidityArr, twinResultsPath, T, distFun)


def createLABELS(T):
    labels = [" "] * T
    labels[0] = str(1)
    labels[T - 1] = str(T)
    labels[int(T / 2) - 1] = str(T / 2)
    return labels


def plotRankOrderStability2(priorE0Arr, cueValidityArr, twinResultsPath, T, distFun):
    """
    :param priorE0Arr:
    :param cueValidityArr:
    :param twinResultsPath:
    :param T:
    :param distFun:
    :return:
    """

    '''
    We cannot use a correlation coefficient to determine rank-order stability because there might be cases in which 
    there is no variability in ranks 
    '''

    # first open the dictionary containing the results
    # for prior, cue reliability combination it contains a matrix with the ranks across time steps
    results = pickle.load(open(os.path.join(twinResultsPath, "rankOrderStabilityRanks.p"), "rb"))

    # what do we want to plot?
    # could have a plot with the correlation coefficient between consecutive timesteps
    # or a whole correlation matrix, heatplot? start with this
    # want to represent the proportion of ties as well

    fig, axes = plt.subplots(len(cueValidityArr), len(priorE0Arr), sharex=True, sharey=True)
    plt.subplots_adjust(top=0.92, bottom=0.12)
    specialAx = fig.add_axes([.16, .040, .7, .01])
    fig.set_size_inches(16, 16)
    ax_list = fig.axes
    simRange = []
    for cueVal in cueValidityArr:
        for pE0 in priorE0Arr:
            current = results[(pE0, cueVal)]
            rankMatrix = current[0]
            probabilities = current[1]

            rankDf = pd.DataFrame(rankMatrix)
            rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(axis=0)] + 0.1

            if distFun == 'cosine':
                sim = cosine_similarity(rankDf.transpose())
            elif distFun == "negativeSwitches":
                sim = calcNegativeRankSwitches(rankDf, T, 'unstable', probabilities)

            simRange += list(sim.flatten())

    boundary1 = min(simRange)
    boundary2 = max(simRange)

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]
            plt.sca(ax)
            # loading the ranks for the current prior - cue reliability combination
            current = results[(pE0, cueVal)]
            rankMatrix = current[0]
            probabilities = current[1]

            rankDf = pd.DataFrame(rankMatrix)  # convert to pandas dataframe for convenience
            # add a small increment to columns that contain only zero entries, otherwise cosine similarity is not defined
            rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(
                axis=0)] + 0.1  # returns columns that are all zeros

            # calculating the similarity matrix
            if distFun == 'cosine':
                sim = cosine_similarity(rankDf.transpose())
                cmap = 'YlGnBu'
                yLabel = 'Cosine similarity'
            elif distFun == "negativeSwitches":
                sim = calcNegativeRankSwitches(rankDf, T, 'unstable', probabilities)
                cmap = 'Greys'  # 'YlGnBu'
                yLabel = 'ontogeny'

            # only negative rank switches

            # create a mask for the upper triangle
            mask = np.tri(sim.shape[0], k=0)

            if jx == len(priorE0Arr) - 1 and ix == 0:
                cbar = True
                cbar_ax = specialAx
                cbar_kws = {"orientation": 'horizontal', "fraction": 0.15, "pad": 0.15,
                            'label': "proportion of rank switches"}  # 'label':"Proportion of negative rank switches",
                sns.heatmap(sim,
                            xticklabels=createLABELS(T),
                            yticklabels=createLABELS(T), vmin=boundary1 - 0.05, vmax=boundary2, cmap=cmap, mask=mask,
                            cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws)

                cbar = ax.collections[0].colorbar
                # here set the labelsize by 20
                cbar.ax.tick_params(labelsize=15)
                cbar.ax.xaxis.label.set_size(20)

                ax2 = ax.twinx()
                ax2.bar(np.arange(1, T, 1), np.diag(sim, 1), linewidth=3, color='k', align='center', width=0.8)


            else:
                cbar = False
                cbar_ax = None
                cbar_kws = None
                sns.heatmap(sim,
                            xticklabels=createLABELS(T),
                            yticklabels=createLABELS(T), vmin=boundary1 - 0.05, vmax=boundary2, cmap=cmap, mask=mask,
                            cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws)
                ax.tick_params(labelsize=15)

                ax2 = ax.twinx()
                ax2.bar(np.arange(1, T, 1), np.diag(sim, 1), linewidth=3, color='k', align='edge', width=0.8)

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax2.set_ylim(0, 1)
            ax2.get_xaxis().tick_bottom()
            ax2.get_yaxis().tick_right()

            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)

            if ix == 0:
                plt.title(str(1 - pE0), fontsize=20)

            if ix == len(cueValidityArr) - 1 and jx == 0:
                ax.set_xlabel('ontogeny', fontsize=20, labelpad=15)
                ax.yaxis.set_label_position("left")
                ax.set_ylabel('%s' % yLabel, fontsize=20, labelpad=15)

                ax2.set_yticks(np.arange(0, 1.1, 0.2))
                ax2.tick_params(labelsize=15)
            else:
                # ax.get_xaxis().set_visible(False)
                ax2.set_yticks([])

            # if jx == 0:
            #     ax.yaxis.set_label_position("left")
            #     ax.set_ylabel('%s' % yLabel, fontsize=20, labelpad=15)
            #     ax2.set_yticks(np.arange(0, 1.1, 0.2))
            #     ax2.tick_params(labelsize=15)
            # else:
            #     ax2.set_yticks([])

            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=20, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            jx += 1
        ix += 1
        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 rotation='vertical')
    plt.savefig(os.path.join(twinResultsPath, 'rankOrderStability2%s.pdf' % distFun), dpi=1200)
    plt.close()

    # # second plot is for rank stability
    # fig, axes = plt.subplots(len(cueValidityArr), len(priorE0Arr), sharex=True, sharey=True)
    # specialAx = fig.add_axes([.16, .055, .7, .01])
    # fig.set_size_inches(16, 16)
    # ax_list = fig.axes
    # simRange = []
    # for cueVal in cueValidityArr:
    #     for pE0 in priorE0Arr:
    #         rankMatrix = ranks[(pE0, cueVal)]
    #
    #         rankDf = pd.DataFrame(rankMatrix)
    #         rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(axis=0)] + 0.1
    #
    #         if distFun == 'cosine':
    #             sim = cosine_similarity(rankDf.transpose())
    #         elif distFun == "negativeSwitches":
    #             sim = calcNegativeRankSwitches(rankDf, T, "stable")
    #
    #         simRange += list(sim.flatten())
    #
    # boundary1 = min(simRange)
    # boundary2 = max(simRange)
    #
    # ix = 0
    # for cueVal in cueValidityArr:
    #     jx = 0
    #     for pE0 in priorE0Arr:
    #         ax = ax_list[ix * len(priorE0Arr) + jx]
    #         plt.sca(ax)
    #         # loading the ranks for the current prior - cue reliability combination
    #         rankMatrix = ranks[(pE0, cueVal)]
    #
    #         rankDf = pd.DataFrame(rankMatrix)  # convert to pandas dataframe for convenience
    #         # add a small increment to columns that contain only zero entries, otherwise cosine similarity is not defined
    #         rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(
    #             axis=0)] + 0.1  # returns columns that are all zeros
    #
    #         # calculating the similarity matrix
    #         if distFun == 'cosine':
    #             sim = cosine_similarity(rankDf.transpose())
    #             cmap = 'YlGnBu'
    #             yLabel = 'Cosine similarity'
    #         elif distFun == "negativeSwitches":
    #             sim = calcNegativeRankSwitches(rankDf, T, 'stable')
    #             cmap = 'Greys'  # 'YlGnBu'
    #             yLabel = 'Time step'
    #
    #         # only negative rank switches
    #
    #         # create a mask for the upper triangle
    #         mask = np.tri(sim.shape[0], k=0)
    #         if jx == len(priorE0Arr) - 1 and ix == 0:
    #             cbar = True
    #             cbar_ax = specialAx
    #             cbar_kws = {"orientation": 'horizontal', "fraction": 0.15, "pad": 0.15,
    #                         'label': "Proportion of stable ranks"}  # 'label':"Proportion of negative rank switches",
    #             sns.heatmap(sim,
    #                         xticklabels=createLABELS(T),
    #                         yticklabels=createLABELS(T), vmin=boundary1 - 0.05, vmax=boundary2, cmap=cmap, mask=mask,
    #                         cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws)
    #
    #             cbar = ax.collections[0].colorbar
    #             # here set the labelsize by 20
    #             cbar.ax.tick_params(labelsize=14)
    #             cbar.ax.xaxis.label.set_size(20)
    #         else:
    #             cbar = False
    #             cbar_ax = None
    #             cbar_kws = None
    #             sns.heatmap(sim,
    #                         xticklabels=createLABELS(T),
    #                         yticklabels=createLABELS(T), vmin=boundary1 - 0.05, vmax=boundary2, cmap=cmap, mask=mask,
    #                         cbar=cbar, cbar_ax=cbar_ax, cbar_kws=cbar_kws)
    #             ax.tick_params(labelsize=14)
    #
    #         ax.get_xaxis().tick_bottom()
    #         ax.get_yaxis().tick_left()
    #
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         ax.spines['bottom'].set_visible(False)
    #         ax.spines['left'].set_visible(False)
    #
    #         if ix == 0:
    #             plt.title(str(1 - pE0), fontsize=20)
    #
    #         if ix == len(cueValidityArr) - 1:
    #             ax.set_xlabel('Time step', fontsize=20, labelpad=10)
    #         else:
    #             ax.get_xaxis().set_visible(False)
    #
    #         if jx == 0:
    #             ax.yaxis.set_label_position("left")
    #             ax.set_ylabel('%s' % yLabel, fontsize=20, labelpad=10)
    #         if jx == len(priorE0Arr) - 1:
    #             plt.ylabel(str(cueVal), labelpad=20, rotation='vertical', fontsize=20)
    #             ax.yaxis.set_label_position("right")
    #
    #         jx += 1
    #     ix += 1
    #     plt.suptitle('Prior probability', fontsize=20)
    #     fig.text(0.98, 0.5, 'Cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
    #              transform=ax.transAxes, rotation='vertical')
    # plt.savefig(os.path.join(twinResultsPath, 'rankOrderStabilityPos1%s.pdf' % distFun), dpi=1200)
    # plt.close()
    #
    # # 3rd plot
    # fig, axes = plt.subplots(len(priorE0Arr), len(cueValidityArr), sharex=True, sharey=True)
    # fig.set_size_inches(16, 16)
    # ax_list = fig.axes
    # simRange = []
    # for cueVal in cueValidityArr:
    #     for pE0 in priorE0Arr:
    #         rankMatrix = ranks[(pE0, cueVal)]
    #
    #         rankDf = pd.DataFrame(rankMatrix)
    #         rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(axis=0)] + 0.1
    #
    #         if distFun == 'cosine':
    #             sim = cosine_similarity(rankDf.transpose())
    #         elif distFun == "negativeSwitches":
    #             sim = calcNegativeRankSwitches(rankDf, T, 'unstable')
    #
    #         simRange += list(sim.flatten())
    #
    # ix = 0
    # for cueVal in cueValidityArr:
    #     jx = 0
    #     for pE0 in priorE0Arr:
    #         ax = ax_list[ix * len(priorE0Arr) + jx]
    #         plt.sca(ax)
    #         # loading the ranks for the current prior - cue reliability combination
    #         rankMatrix = ranks[(pE0, cueVal)]
    #
    #         rankDf = pd.DataFrame(rankMatrix)  # convert to pandas dataframe for convenience
    #         # add a small increment to columns that contain only zero entries, otherwise cosine similarity is not defined
    #         rankDf.loc[:, (rankDf == 0.0).all(axis=0)] = rankDf.loc[:, (rankDf == 0.0).all(
    #             axis=0)] + 0.1  # returns columns that are all zeros
    #
    #         # calculating the similarity matrix
    #         if distFun == 'cosine':
    #             sim = cosine_similarity(rankDf.transpose())
    #         elif distFun == "negativeSwitches":
    #             sim = calcNegativeRankSwitches(rankDf, T, 'unstable')
    #
    #         if jx == len(priorE0Arr) - 1 and ix == 0:
    #             ax.bar(np.arange(1, T, 1), np.diag(sim, 1), linewidth=3, color='k', align='center', width=0.8)
    #
    #
    #         else:
    #             ax.bar(np.arange(1, T, 1), np.diag(sim, 1), linewidth=3, color='k', align='edge', width=0.8)
    #
    #         ax.get_xaxis().tick_bottom()
    #         ax.get_yaxis().tick_left()
    #
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
    #         ax.spines['bottom'].set_visible(True)
    #         ax.spines['left'].set_visible(False)
    #
    #         ax.set_ylim(0, 1)
    #         plt.yticks([])
    #         plt.xticks([])
    #
    #         if ix == 0:
    #             plt.title(str(1 - pE0), fontsize=20)
    #         #
    #         # if jx == 0:
    #         #     plt.title(str(cueVal), fontsize=30)
    #         #
    #         # if ix == 0 and jx == 0:
    #         #     ax.set_xlabel('Time', fontsize=30, labelpad=10)
    #         #     ax.spines['left'].set_visible(True)
    #         #     ax.yaxis.set_label_position("left")
    #         #     ax.set_ylabel('Proportion of rank switches', fontsize=30, labelpad=10)
    #
    #         if ix == len(cueValidityArr) - 1:
    #             ax.set_xlabel('Time step', fontsize=20, labelpad=10)
    #         else:
    #             ax.get_xaxis().set_visible(False)
    #
    #         if jx == 0:
    #             ax.yaxis.set_label_position("left")
    #             ax.set_ylabel('Proportion of rank switches', fontsize=20, labelpad=10)
    #
    #         if jx == len(priorE0Arr) - 1:
    #             plt.ylabel(str(cueVal), labelpad=20, rotation='vertical', fontsize=20)
    #             ax.yaxis.set_label_position("right")
    #
    #         jx += 1
    #     ix += 1
    #     plt.suptitle('Prior probability', fontsize=20)
    #     fig.text(0.98, 0.5, 'Cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
    #              transform=ax.transAxes, rotation='vertical')
    # plt.savefig(os.path.join(twinResultsPath, 'rankOrderStabilityPos2%s.pdf' % distFun), dpi=1200)
    # plt.close()


def plotBeliefAndPhenotypeDivergence(tValues, priorE0Arr, cueValidityArr, relativeDistanceDict, twinResultsPath,
                                     argument, adoptionType, lag, endOfExposure, beliefDict,
                                     relativeDistanceDictTemp):
    fig, axes = plt.subplots(len(priorE0Arr), len(cueValidityArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]

            plt.sca(ax)

            relativeDistance = relativeDistanceDict[(pE0, cueVal)]

            relativeDistanceDiff = np.gradient(relativeDistance)

            posBeliefDiffNoAverage = beliefDict[(pE0, cueVal)][:,
                                     5]  # measured at the end of ontogeny after the last cue
            posBeliefDiffNoAverageDiff = np.gradient(posBeliefDiffNoAverage)

            plt.plot(tValues[0:], posBeliefDiffNoAverageDiff, color='grey', linestyle='solid', linewidth=2,
                     markersize=5,
                     marker='D',
                     markerfacecolor='grey')

            plt.plot(tValues[0:], relativeDistanceDiff, color='black', linestyle='solid', linewidth=2, markersize=5,
                     marker='o',
                     markerfacecolor='black')  # should be absolute distance

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-0.4, 0.05)
            plt.yticks([-0.3, 0, 0.05], fontsize=15)

            if ix == 0:
                plt.title(str(1 - pE0), fontsize=20)

            if ix == len(cueValidityArr) - 1:
                plt.xticks([], fontsize=15)

            else:
                ax.get_xaxis().set_visible(False)

            if jx == 0 and ix == len(cueValidityArr) - 1:
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                plt.xlabel("ontogeny", fontsize=20, labelpad=10)
                plt.ylabel('gradient of plasticity curves', fontsize=20, labelpad=10)

            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            # plot lines for readeability

            tValNew = np.arange(min(tValues) - 0.5, max(tValues) + 0.5 + 1, 1)
            plt.plot(tValNew, [1] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)
            plt.plot(tValNew, [0] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)

            jx += 1
        ix += 1
        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        if endOfExposure:
            safeStr = "EndOfExposure"
        else:
            safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsPath, '%s_%s_%s_%sPlasticityAndBeliefEndOntogenyDivergence.pdf' % (
            argument, adoptionType, lag, safeStr)),
        dpi=1200)
    plt.close()


def plotBeliefDistances(tValues, priorE0Arr, cueValidityArr, relativeDistanceDict, twinResultsPath,
                        argument, adoptionType, lag, endOfExposure, beliefDict,
                        relativeDistanceDictTemp):
    fig, axes = plt.subplots(len(priorE0Arr), len(cueValidityArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]

            plt.sca(ax)

            relativeDistance = relativeDistanceDict[(pE0, cueVal)]

            posBeliefDiffNoAverage = beliefDict[(pE0, cueVal)][:,
                                     5]  # measured at the end of ontogeny after the last cue

            plt.bar(tValues, posBeliefDiffNoAverage, linewidth=3, color='lightgray', align='center', width=0.8)

            plt.plot(tValues, relativeDistance, color='black', linestyle='solid', linewidth=2, markersize=8,
                     marker='o',
                     markerfacecolor='black')  # should be absolute distance

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-0.05, 1.05)
            plt.yticks(np.arange(0, 1.1, 0.2), fontsize=15)

            if ix == 0:
                plt.title(str(1 - pE0), fontsize=20)

            if ix == len(cueValidityArr) - 1:
                plt.xticks([], fontsize=15)

            else:
                ax.get_xaxis().set_visible(False)

            if jx == 0 and ix == len(cueValidityArr) - 1:
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                plt.xlabel("ontogeny", fontsize=20, labelpad=10)
                plt.ylabel('divergence between twins', fontsize=20, labelpad=10)

            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            # plot lines for readeability

            tValNew = np.arange(min(tValues) - 0.5, max(tValues) + 0.5 + 1, 1)
            plt.plot(tValNew, [1] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)
            plt.plot(tValNew, [0] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)

            jx += 1
        ix += 1
        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        if endOfExposure:
            safeStr = "EndOfExposure"
        else:
            safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsPath,
                     '%s_%s_%s_%sPlasticityAndBeliefEndOntogeny.pdf' % (argument, adoptionType, lag, safeStr)),
        dpi=1200)
    plt.close()

    fig, axes = plt.subplots(len(priorE0Arr), len(cueValidityArr), sharex=True, sharey=True)
    fig.set_size_inches(16, 16)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]

            plt.sca(ax)

            posBeliefDiffNoAverage = beliefDict[(pE0, cueVal)][:,
                                     6]  # measured after each cue

            plt.bar(tValues, posBeliefDiffNoAverage, linewidth=3, color='lightgray', align='center', width=0.8)

            relativeDistanceTemp = relativeDistanceDictTemp[(pE0, cueVal)]
            plt.plot(tValues, relativeDistanceTemp, color='black', linestyle='solid', linewidth=2, markersize=8,
                     marker='o', markerfacecolor='black')

            print("The current prior is %s and the cue reliability is %s" % ((1 - pE0), cueVal))
            print("The correlation between information and phenotype divergence is: " + str(
                stats.pearsonr(relativeDistanceTemp, posBeliefDiffNoAverage)[0]))

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            plt.ylim(-0.05, 1.05)
            plt.yticks(np.arange(0, 1.1, 0.2), fontsize=15)

            if ix == 0:
                plt.title(str(1 - pE0), fontsize=20)

            if ix == len(cueValidityArr) - 1:
                plt.xticks([], fontsize=15)

            else:
                ax.get_xaxis().set_visible(False)

            if jx == 0 and ix == len(cueValidityArr) - 1:
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                plt.xlabel("ontogeny", fontsize=20, labelpad=10)
                plt.ylabel('divergence between twins', fontsize=20, labelpad=10)

            if jx == len(priorE0Arr) - 1:
                plt.ylabel(str(cueVal), labelpad=30, rotation='vertical', fontsize=20)
                ax.yaxis.set_label_position("right")

            # plot lines for readeability
            tValNew = np.arange(min(tValues) - 0.5, max(tValues) + 0.5 + 1, 1)
            plt.plot(tValNew, [1] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)
            plt.plot(tValNew, [0] * len(tValNew), ls='--', lw=0.8, color='black', zorder=2)

            jx += 1
        ix += 1
        plt.suptitle('prior probability', fontsize=20)
        fig.text(0.98, 0.5, 'cue reliability', fontsize=20, horizontalalignment='right', verticalalignment='center',
                 transform=ax.transAxes, rotation='vertical')
        if endOfExposure:
            safeStr = "EndOfExposure"
        else:
            safeStr = "EndOfOntogeny"
    plt.savefig(
        os.path.join(twinResultsPath,
                     '%s_%s_%s_%sPlasticityAndBeliefAfterCue.pdf' % (argument, adoptionType, lag, safeStr)),
        dpi=1200)
    plt.close()


def plotDistances(tValues, priorE0Arr, cueValidityArr, absoluteDistanceDict, relativeDistanceDict,
                  twinResultsPath,
                  argument, adoptionType, lag, endOfExposure, VarArg, absoluteDistanceDictVar, relativeDistanceDictVar,
                  env):
    arg = "absolute"  # choose whether you want this plot for relative or absolute distance

    linestyle_tuple = [
        ('solid', (0, ())),
        ('dotted', (0, (1, 1))),
        ('densely dashed', (0, (5, 1))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),
        ('loosely dotted', (0, (1, 10))),
        ('loosely dashed', (0, (5, 10))),
        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('densely dotted', (0, (1, 1))),
        ('dashed', (0, (5, 5))),
        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)))]

    fig, axes = plt.subplots(len(cueValidityArr), len(priorE0Arr), sharex=True, sharey=True)
    fig.set_size_inches(16.5, 16.5)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityArr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]
            ax.set(aspect=max(tValues) - 1)
            plt.sca(ax)
            absoluteDistance = absoluteDistanceDict[(pE0, cueVal)]
            relativeDistance = relativeDistanceDict[(pE0, cueVal)]

            if VarArg:
                # absoluteDistanceVar = absoluteDistanceDictVar[(markov_chain, cueVal)]
                # plt.plot(tValues, absoluteDistance, color='grey', linestyle='solid', linewidth=2, markersize=8,
                #          marker='D',
                #          markerfacecolor='grey')
                # plt.errorbar(tValues, absoluteDistance, yerr=absoluteDistanceVar, fmt="none", ecolor='grey')

                relativeDistanceVar = relativeDistanceDictVar[(pE0, cueVal)]

                plt.plot(tValues, relativeDistance, color='black', linestyle='--', linewidth=2, markersize=8,
                         marker='o', markerfacecolor='black')
                plt.errorbar(tValues, relativeDistance, yerr=relativeDistanceVar, fmt="none", ecolor='black')

            else:

                if arg == 'absolute':
                    [plt.plot(tValues, keyVal[1], linestyle=linestyle_tuple[idx][1], linewidth=2, markersize=5,
                              marker='o', color='black', markerfacecolor='black',
                              label="%s" % (keyVal[0])) for idx, keyVal in enumerate(absoluteDistance.items())]
                else:
                    for idx, keyVal in enumerate(relativeDistance.items()):
                        # plt.plot(tValues, keyVal[1], linestyle=linestyle_tuple[idx][1], linewidth=2, markersize=5,
                        #           marker='o', color='black', markerfacecolor='black',
                        #           label="%s" % (keyVal[0]))
                        #This might be the best alternative to properly represent the phenotypes
                        if keyVal[0] != 'y0y1':
                            plt.plot(tValues, keyVal[1], linestyle=linestyle_tuple[idx][1], linewidth=2, markersize=5,
                                      marker='o', color='black', markerfacecolor='black',
                                      label="%s" % (keyVal[0]))
                        else:

                            plt.plot(tValues, absoluteDistance['y0y1'], linestyle=linestyle_tuple[idx][1], linewidth=2, markersize=5,
                                      marker='o', color='black', markerfacecolor='black',
                                      label="%s" % ('y0y1'))

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)

            plt.ylim(-0.05, 1.05)
            plt.yticks(np.arange(0, 1.1, 0.2), fontsize=15)
            plt.xticks(np.arange(2, max(tValues) + 1, 2), fontsize=15)

            if ix == len(cueValidityArr) - 1 and jx == 0:
                anchPar = len(priorE0Arr) / float(2)
                legend = ax.legend(loc='upper center', bbox_to_anchor=(anchPar, -0.2),
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
    plt.savefig(
        os.path.join(twinResultsPath,
                     '%s_%s_%s_%s_%sPlasticity%s_%s.png' % (argument, adoptionType, lag, safeStr, env, VarArg,arg)),
        dpi=600)

    plt.close()


def plotTraces(tValues, priorE0Arr, cueValidityC0E0Arr, atypicalTraceOrgDict, atypicalTraceCloneDict,
                   typicalTraceOrgDict, typicalTraceCloneDict,
                   twinResultsPath, argument, adoptionType, lag, endOfExposure, env):

    arg = 'atypical'
    linestyle_tuple = [
        ('solid', (0, ())),
        ('dotted', (0, (1, 1))),
        ('densely dashed', (0, (5, 1))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),
        ('loosely dotted', (0, (1, 10))),
        ('loosely dashed', (0, (5, 10))),
        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('densely dotted', (0, (1, 1))),
        ('dashed', (0, (5, 5))),
        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)))]

    fig, axes = plt.subplots(len(cueValidityC0E0Arr), len(priorE0Arr), sharex=True, sharey=True)
    fig.set_size_inches(16.5, 16.5)
    ax_list = fig.axes

    ix = 0
    for cueVal in cueValidityC0E0Arr:
        jx = 0
        for pE0 in priorE0Arr:
            ax = ax_list[ix * len(priorE0Arr) + jx]
            ax.set(aspect=0.9)
            plt.sca(ax)
            atypicalTraceOrg = atypicalTraceOrgDict[(pE0, cueVal)]
            atypicalTraceClone = atypicalTraceCloneDict[(pE0, cueVal)]
            typicalTraceOrg = typicalTraceOrgDict[(pE0, cueVal)]
            typicalTraceClone = typicalTraceCloneDict[(pE0, cueVal)]

            if arg == 'atypical':
                # [plt.plot(tValues, keyVal[1], linestyle=linestyle_tuple[idx][1], linewidth=2, markersize=5,
                #           marker='o', color='black', markerfacecolor='black',
                #           label="%s" % (keyVal[0])) for idx, keyVal in enumerate(atypicalTraceOrg.items()) if keyVal[0] != 'y0y1']
                #
                # [plt.plot(tValues, keyVal[1], linestyle=linestyle_tuple[idx][1], linewidth=2, markersize=5,
                #           marker='o', color='grey', markerfacecolor='grey') for idx, keyVal in enumerate(atypicalTraceClone.items()) if keyVal[0] != 'y0y1']


                #experiment
                [plt.plot(tValues, keyVal[1], linestyle=linestyle_tuple[idx][1], linewidth=2, markersize=5,
                          marker='o', color='black', markerfacecolor='black',
                          label="%s" % (keyVal[0])) for idx, keyVal in enumerate(atypicalTraceOrg.items()) if
                 keyVal[0] == 'y0Cy1C']

                [plt.plot(tValues, keyVal[1], linestyle=linestyle_tuple[idx][1], linewidth=2, markersize=5,
                          marker='o', color='black', markerfacecolor='black',
                          label="%s" % (keyVal[0])) for idx, keyVal in enumerate(typicalTraceOrg.items()) if
                 keyVal[0] == 'y0Dy1D']

                [plt.plot(tValues, keyVal[1], linestyle=linestyle_tuple[idx][1], linewidth=2, markersize=5,
                          marker='o', color='grey', markerfacecolor='grey') for idx, keyVal in
                 enumerate(atypicalTraceClone.items()) if keyVal[0] == 'y0Cy1C']

                [plt.plot(tValues, keyVal[1], linestyle=linestyle_tuple[idx][1], linewidth=2, markersize=5,
                          marker='o', color='grey', markerfacecolor='grey') for idx, keyVal in
                 enumerate(typicalTraceClone.items()) if keyVal[0] == 'y0Dy1D']

                ylabel = 'atypical specializations & \n typical deconstructions'

            else:
                [plt.plot(tValues, keyVal[1], linestyle=linestyle_tuple[idx][1], linewidth=2, markersize=5,
                          marker='o', color='black', markerfacecolor='black',
                          label="%s" % (keyVal[0])) for idx, keyVal in enumerate(typicalTraceOrg.items()) if keyVal[0] != 'y0y1']
                [plt.plot(tValues, keyVal[1], linestyle=linestyle_tuple[idx][1], linewidth=2, markersize=5,
                          marker='o', color='grey', markerfacecolor='grey',
                          label="%s" % (keyVal[0])) for idx, keyVal in enumerate(typicalTraceClone.items()) if keyVal[0] != 'y0y1']

                ylabel = 'typical specializations & \n deconstructions'

            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)

            plt.ylim(-0.5, max(tValues)+0.5)
            plt.yticks(np.arange(0, max(tValues)+1, 2), fontsize=15)
            plt.xticks(np.arange(2, max(tValues) + 1, 2), fontsize=15)

            if ix == len(cueValidityC0E0Arr) - 1 and jx == 0:
                anchPar = len(priorE0Arr) / float(2)
                legend = ax.legend(loc='upper center', bbox_to_anchor=(anchPar, -0.2),
                                   title='phenotypic dimensions',
                                   ncol=3, fancybox=True, shadow=False, fontsize=20)
                plt.setp(legend.get_title(), fontsize='20')
            if ix == 0:
                plt.title("%s" % (1 - pE0), fontsize=20)

            if (ix == (len(cueValidityC0E0Arr) - 1)) and (jx == 0):
                plt.xlabel("ontogeny", fontsize=20, labelpad=10)
                plt.ylabel(ylabel, fontsize=20, labelpad=10)

            if ix < (len(cueValidityC0E0Arr) - 1):
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
    plt.savefig(
        os.path.join(twinResultsPath,
                     '%s_%s_%s_%s_%sTraces%s.pdf' % (argument, adoptionType, lag, safeStr, env, arg)),
        dpi=600)

    plt.close()


def postProcessTraces(
        atypicalTraces, typicalTraces, cueProbabilities):
    atypicalTraceOrg = {}
    atypicalTraceClone = {}
    typicalTraceOrg = {}
    typicalTraceClone = {}

    # store for each key
    listOfKeys = ['y0y1', 'y0Cy1C', 'y0Dy1D']
    for elem in listOfKeys:
        atypicalTraceOrg[elem] = []
        atypicalTraceClone[elem] = []
        typicalTraceOrg[elem] = []
        typicalTraceClone[elem] = []

    # for atypical and typical traces we store the following info:
    # for each type of specializations and each time step: a list the length of which corresponds to the number of agents
    # one element is (original # specialized, clone # specialized)
    for (key, tPeriod), val in atypicalTraces.items():
        dataOrg, dataClone = zip(*val)
        atypicalTraceOrg[key].append(np.average(dataOrg, weights=cueProbabilities))
        atypicalTraceClone[key].append(np.average(dataClone, weights=cueProbabilities))

    for (key, tPeriod), val in typicalTraces.items():
        dataOrg, dataClone = zip(*val)
        typicalTraceOrg[key].append(np.average(dataOrg, weights=cueProbabilities))
        typicalTraceClone[key].append(np.average(dataClone, weights=cueProbabilities))

    return atypicalTraceOrg, atypicalTraceClone, typicalTraceOrg, typicalTraceClone


def performSimulationAnalysis(argument, priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath, numAgents,
                              resultsPath, baselineFitness, argumentR, argumentP, lag, adoptionType, endOfExposure,
                              env):
    # first step create the directory for the results
    if not os.path.exists(twinResultsPath):
        os.makedirs(twinResultsPath)

    if argument == "ExperimentalTwinstudy":
        """
        this will implement a form of the twin study that an be considered more artifical
        it will be comparable to experimental manipulations done in lab environments
        it will manipulate the onset and amount of a
        """

        absoluteDistanceDict = {}
        relativeDistanceDict = {}

        # for traces
        atypicalTraceOrgDict = {}
        atypicalTraceCloneDict = {}
        typicalTraceOrgDict = {}
        typicalTraceCloneDict = {}

        for pE0 in priorE0Arr:
            for cueReliability in cueValidityC0E0Arr:
                print("currently working with pE0: " + str(pE0) + " and cue reliability: " + str(cueReliability))

                startTime = time.perf_counter()

                """
                need to extract the correct prior from the markov chain
                """

                # resultsMat already contains euclidean distances
                resultsMat, cueProbabilities, atypicalTraces, typicalTraces = runExperimentalAdoptionExperiment(T, numAgents, env, pE0,
                                                                                 cueReliability,
                                                                                 resultsPath,
                                                                                 argumentR, argumentP, lag,
                                                                                 adoptionType, endOfExposure)
                elapsedTime = time.perf_counter() - startTime
                print("Elapsed time: " + str(elapsedTime))

                # postprocessing is used to normalize the euclidean distance to a range form 0 to 1

                absoluteDistance, relativeDistance, _, _ = postProcessResultsMat(resultsMat, T + lag - 1, endOfExposure,
                                                                                 lag, cueProbabilities)
                absoluteDistanceDict[(pE0, cueReliability)] = absoluteDistance
                relativeDistanceDict[(pE0, cueReliability)] = relativeDistance


                # traces
                atypicalTraceOrg, atypicalTraceClone, typicalTraceOrg, typicalTraceClone = postProcessTraces(
                    atypicalTraces, typicalTraces, cueProbabilities)
                # store traces
                atypicalTraceOrgDict[(pE0, cueReliability)] = atypicalTraceOrg
                atypicalTraceCloneDict[(pE0, cueReliability)] = atypicalTraceClone
                typicalTraceOrgDict[(pE0, cueReliability)] = typicalTraceOrg
                typicalTraceCloneDict[(pE0, cueReliability)] = typicalTraceClone

        pickle.dump(absoluteDistanceDict, open(os.path.join(twinResultsPath, 'absoluteDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))
        pickle.dump(relativeDistanceDict, open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))

        # save the traces locally
        # atypicalTraceOrg, atypicalTraceClone,typicalTraceOrg, typicalTraceClone
        pickle.dump(atypicalTraceOrgDict, open(os.path.join(twinResultsPath, 'atypicalTraceOrgDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))
        pickle.dump(atypicalTraceCloneDict, open(os.path.join(twinResultsPath, 'atypicalTraceCloneDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))
        pickle.dump(typicalTraceOrgDict, open(os.path.join(twinResultsPath, 'typicalTraceOrgDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))
        pickle.dump(typicalTraceCloneDict, open(os.path.join(twinResultsPath, 'typicalTraceCloneDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))

    elif argument == "Twinstudy":
        """
        This will calculate the results from the twin studies
        """

        absoluteDistanceDict = {}
        relativeDistanceDict = {}
        absoluteDistanceDictVar = {}
        relativeDistanceDictVar = {}
        resultsMatTwinStudyDict = {}

        # for traces
        atypicalTraceOrgDict = {}
        atypicalTraceCloneDict = {}
        typicalTraceOrgDict = {}
        typicalTraceCloneDict = {}

        beliefDict = {}
        for pE0 in priorE0Arr:
            for cueReliability in cueValidityC0E0Arr:
                print("currently working with prior: " + str(pE0) + " and cue reliability: " + str(cueReliability))

                startTime = time.perf_counter()
                resultsMat, resultsMatBeliefs, resultsMatTempPhenotypes, cueProbabilities, atypicalTraces, typicalTraces = runAdoptionExperiment(
                    T,
                    numAgents,
                    env,
                    pE0,
                    cueReliability,
                    resultsPath,
                    argumentR,
                    argumentP,
                    adoptionType)
                elapsedTime = time.perf_counter() - startTime
                print("Elapsed time: " + str(elapsedTime))

                # normalize resultsmat
                resultsMatTwinStudyDict[((pE0, cueReliability))] = resultsMat

                pickle.dump(cueReliability, open(os.path.join(twinResultsPath,
                                                              "cueProbability_%s_%s_%s.p" % (
                                                                  pE0, env, cueReliability)),
                                                 "wb"))

                # traces
                atypicalTraceOrg, atypicalTraceClone, typicalTraceOrg, typicalTraceClone = postProcessTraces(
                    atypicalTraces, typicalTraces, cueProbabilities)

                # phenotypic distance
                absoluteDistance, relativeDistance, absoluteDistanceVariance, relativeDistanceVariance = postProcessResultsMat(
                    resultsMat, T, endOfExposure, lag, cueProbabilities)

                absoluteDistanceDict[(pE0, cueReliability)] = absoluteDistance
                relativeDistanceDict[(pE0, cueReliability)] = relativeDistance

                # store traces
                atypicalTraceOrgDict[(pE0, cueReliability)] = atypicalTraceOrg
                atypicalTraceCloneDict[(pE0, cueReliability)] = atypicalTraceClone
                typicalTraceOrgDict[(pE0, cueReliability)] = typicalTraceOrg
                typicalTraceCloneDict[(pE0, cueReliability)] = typicalTraceClone

                absoluteDistanceDictVar[(pE0, cueReliability)] = absoluteDistanceVariance
                relativeDistanceDictVar[(pE0, cueReliability)] = relativeDistanceVariance

                beliefDict[(pE0, cueReliability)] = resultsMatBeliefs

        pickle.dump(resultsMatTwinStudyDict,
                    open(os.path.join(twinResultsPath,
                                      "resultsMat%s%s%s%s_%s.p" % (argument, adoptionType, lag, endOfExposure, env)),
                         "wb"))
        pickle.dump(absoluteDistanceDict, open(os.path.join(twinResultsPath, 'absoluteDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))
        pickle.dump(relativeDistanceDict, open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))

        # save the traces locally
        # atypicalTraceOrg, atypicalTraceClone,typicalTraceOrg, typicalTraceClone
        pickle.dump(atypicalTraceOrgDict, open(os.path.join(twinResultsPath, 'atypicalTraceOrgDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))
        pickle.dump(atypicalTraceCloneDict, open(os.path.join(twinResultsPath, 'atypicalTraceCloneDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))
        pickle.dump(typicalTraceOrgDict, open(os.path.join(twinResultsPath, 'typicalTraceOrgDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))
        pickle.dump(typicalTraceCloneDict, open(os.path.join(twinResultsPath, 'typicalTraceCloneDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))

        pickle.dump(absoluteDistanceDictVar,
                    open(os.path.join(twinResultsPath, 'absoluteDistanceDictVar%s%s%s%s_%s.p' % (
                        argument, adoptionType, lag, endOfExposure, env)), 'wb'))
        pickle.dump(relativeDistanceDictVar,
                    open(os.path.join(twinResultsPath, 'relativeDistanceDictVar%s%s%s%s_%s.p' % (
                        argument, adoptionType, lag, endOfExposure, env)), 'wb'))

        pickle.dump(beliefDict, open(os.path.join(twinResultsPath, 'beliefsDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'wb'))

        del absoluteDistanceDict
        del relativeDistanceDict
        del absoluteDistanceDictVar
        del relativeDistanceDictVar
        del beliefDict
        del resultsMatTwinStudyDict

    elif argument == "MaturePhenotypes":
        maturePhenotypes = {}
        maturePhenotypesTemp = {}
        for pE0 in priorE0Arr:
            for cueReliability in cueValidityC0E0Arr:
                print("Currently working on prior: " + str(pE0) + " and cue reliability: " + str(cueReliability))
                maturePheno, maturePhenotypesTemporal, _, cueProbabilities = \
                    runTwinStudiesParallel(0, numAgents, env, pE0,
                                           cueReliability, False, T,
                                           resultsPath, argumentR, argumentP,
                                           None)

                maturePhenotypes[(pE0, cueReliability)] = maturePheno, cueProbabilities
                maturePhenotypesTemp[(pE0, cueReliability)] = maturePhenotypesTemporal, cueProbabilities

        pickle.dump(maturePhenotypes, open(os.path.join(twinResultsPath, "maturePhenotypes_%s.p" % (env)), "wb"))
        pickle.dump(maturePhenotypesTemp,
                    open(os.path.join(twinResultsPath, "maturePhenotypesTemp_%s.p" % (env)), "wb"))
        del maturePhenotypes
        del maturePhenotypesTemp

    elif argument == "RankOrderStability":
        """
        for now this argument does not work with the changing environments because we have not agreed what the trait 
        of interest is when the environment fluctuates; best average fit with the environment across adulthood? 

        but is this really a single trait? I would leave it be for now 
        """

        rankOrderStabilityRaw = {}
        rankOrderStabilityRanks = {}

        for pE0 in priorE0Arr:
            for cueReliability in cueValidityC0E0Arr:
                print("Currently working on prior: " + str(pE0) + " and cue reliability: " + str(cueReliability))

                maturePheno, maturePhenotypesTemporal, _, cueProbabilities = \
                    runTwinStudiesParallel(0, numAgents, env, pE0,
                                           cueReliability, False, T,
                                           resultsPath, argumentR,
                                           argumentP,
                                           None)  # uses the temp phenotypes

                rankOrderStabilityRaw[(pE0, cueReliability)] = maturePhenotypesTemporal
                current = rankOrderStabilityRaw[(pE0, cueReliability)]
                currentMat = np.zeros((current.shape[0], T))

                tValues = np.arange(1, T + 1, 1)
                for t in tValues:
                    possibleRanks = sorted(list(set(current[:, 1, t - 1])), reverse=True)
                    currentMat[:, t - 1] = [possibleRanks.index(a) + 1 for a in current[:, 1,
                                                                                t - 1]]  # the plus one makes sure that we don't have zero ranks, which are computationally inconvenient

                rankOrderStabilityRanks[(pE0, cueReliability)] = (currentMat, cueProbabilities)
        # TODO add the env argument just in case
        pickle.dump(rankOrderStabilityRaw,
                    open(os.path.join(twinResultsPath, "rankOrderStabilityRaw.p"), "wb"))
        pickle.dump(rankOrderStabilityRanks,
                    open(os.path.join(twinResultsPath, "rankOrderStabilityRanks.p"), "wb"))

    elif argument == 'FitnessDifference':
        fitnessDifference(priorE0Arr, cueValidityC0E0Arr, resultsPath, T, twinResultsPath, baselineFitness,
                          argumentR,
                          argumentP, numAgents)


    else:
        print("Wrong input argument to plotting arguments!")
        exit()


def plotSimulationStudy(argument, priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath, lag, adoptionType,
                        endOfExposure, varArg, env):
    tValues = np.arange(1, T + 1, 1)

    if argument == "BeliefTwinstudy":
        relativeDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s_%s.p' % (
            "Twinstudy", adoptionType, lag, endOfExposure, env)), 'rb'))

        # for the temporary phenotypes
        relativeDistanceDictTemp = pickle.load(
            open(os.path.join(twinResultsPath, 'relativeDistanceDictTemp%s%s%s%s_%s.p' % (
                "Twinstudy", adoptionType, lag, endOfExposure, env)), 'rb'))

        beliefDict = pickle.load(open(os.path.join(twinResultsPath, 'beliefsDict%s%s%s%s_%s.p' % (
            "Twinstudy", adoptionType, lag, endOfExposure, env)), 'rb'))

        plotBeliefDistances(tValues, priorE0Arr, cueValidityC0E0Arr, relativeDistanceDict, twinResultsPath,
                            argument, adoptionType, lag, endOfExposure, beliefDict,
                            relativeDistanceDictTemp, env)

    elif argument == "Twinstudy":

        absoluteDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'absoluteDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'rb'))
        relativeDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'rb'))

        # load the variance
        if varArg:
            absoluteDistanceDictVar = pickle.load(
                open(os.path.join(twinResultsPath, 'absoluteDistanceDictVar%s%s%s%s_%s.p' % (
                    argument, adoptionType, lag, endOfExposure, env)), 'rb'))
            relativeDistanceDictVar = pickle.load(
                open(os.path.join(twinResultsPath, 'relativeDistanceDictVar%s%s%s%s_%s.p' % (
                    argument, adoptionType, lag, endOfExposure, env)), 'rb'))

        else:
            absoluteDistanceDictVar = None
            relativeDistanceDictVar = None

        plotDistances(tValues, priorE0Arr, cueValidityC0E0Arr, absoluteDistanceDict, relativeDistanceDict,
                      twinResultsPath, argument, adoptionType, lag, endOfExposure, varArg, absoluteDistanceDictVar,
                      relativeDistanceDictVar, env)

        # new addition:
        # for original and clone plot the number of time specialized towards the less likely environment
        # for original and clone plot the number of time specialializations towards the less likely environment were deconstructed


        atypicalTraceOrgDict = pickle.load(open(os.path.join(twinResultsPath, 'atypicalTraceOrgDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'rb'))
        atypicalTraceCloneDict = pickle.load(
            open(os.path.join(twinResultsPath, 'atypicalTraceCloneDict%s%s%s%s_%s.p' % (
                argument, adoptionType, lag, endOfExposure, env)), 'rb'))
        typicalTraceOrgDict = pickle.load(open(os.path.join(twinResultsPath, 'typicalTraceOrgDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'rb'))
        typicalTraceCloneDict = pickle.load(open(os.path.join(twinResultsPath, 'typicalTraceCloneDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'rb'))

        plotTraces(tValues, priorE0Arr, cueValidityC0E0Arr, atypicalTraceOrgDict, atypicalTraceCloneDict,
                   typicalTraceOrgDict, typicalTraceCloneDict,
                   twinResultsPath, argument, adoptionType, lag, endOfExposure, env)

    elif argument == "ExperimentalTwinstudy":
        absoluteDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'absoluteDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'rb'))
        relativeDistanceDict = pickle.load(open(os.path.join(twinResultsPath, 'relativeDistanceDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'rb'))

        plotDistances(tValues, priorE0Arr, cueValidityC0E0Arr, absoluteDistanceDict, relativeDistanceDict,
                      twinResultsPath, argument, adoptionType, lag, endOfExposure, False, None, None, env)


        atypicalTraceOrgDict = pickle.load(open(os.path.join(twinResultsPath, 'atypicalTraceOrgDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'rb'))
        atypicalTraceCloneDict = pickle.load(
            open(os.path.join(twinResultsPath, 'atypicalTraceCloneDict%s%s%s%s_%s.p' % (
                argument, adoptionType, lag, endOfExposure, env)), 'rb'))
        typicalTraceOrgDict = pickle.load(open(os.path.join(twinResultsPath, 'typicalTraceOrgDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'rb'))
        typicalTraceCloneDict = pickle.load(open(os.path.join(twinResultsPath, 'typicalTraceCloneDict%s%s%s%s_%s.p' % (
            argument, adoptionType, lag, endOfExposure, env)), 'rb'))

        plotTraces(tValues, priorE0Arr, cueValidityC0E0Arr, atypicalTraceOrgDict, atypicalTraceCloneDict,
                   typicalTraceOrgDict, typicalTraceCloneDict,
                   twinResultsPath, argument, adoptionType, lag, endOfExposure, env)

    elif argument == "MaturePhenotypes":
        print("loading mature phenotypes")
        maturePhenotypesResults = pickle.load(
            open(os.path.join(twinResultsPath, "maturePhenotypes_%s.p" % (env)), "rb"))

        plotTriangularPlots(tValues, priorE0Arr, cueValidityC0E0Arr, maturePhenotypesResults, T,
                            twinResultsPath, env)

        plotTriangularPlotsArrow(tValues, priorE0Arr, cueValidityC0E0Arr, maturePhenotypesResults, T,
                            twinResultsPath, env)


    elif argument == "MaturePhenotypesTwoPatches":
        maturePhenotypes = pickle.load(
            open(os.path.join(twinResultsPath, "maturePhenotypesTwoPatches_%s.p" % (env)), "rb"))
        plotTriangularPlots(tValues, priorE0Arr, cueValidityC0E0Arr, maturePhenotypes, T, twinResultsPath)

    elif argument == "FitnessDifference":
        plotFitnessDifference(priorE0Arr, cueValidityC0E0Arr, twinResultsPath)

    elif argument == "RankOrderStability":
        plotRankOrderStability(priorE0Arr, cueValidityC0E0Arr, twinResultsPath, T, ['negativeSwitches'])

    else:
        print("Wrong input argument to plotting arguments!")


def runPlots(priorE0Arr, cueValidityC0E0Arr, TParam, numAgents, twinResultsPath, baselineFitness, resultsPath,
             argumentR, argumentP, lagArray, adoptionType, endOfExposure, plotArgs, plotVar, performSimulation,
             startingEnv):
    for arg in plotArgs:
        if arg == 'ExperimentalTwinstudy':
            for lag in lagArray:
                TLag = TParam - lag + 1
                T = TLag
                if performSimulation:
                    performSimulationAnalysis(arg, priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath,
                                              numAgents,
                                              resultsPath, baselineFitness, argumentR, argumentP, lag, adoptionType,
                                              endOfExposure, startingEnv)
                plotSimulationStudy(arg, priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath, lag, adoptionType,
                                    endOfExposure, plotVar, startingEnv)
        elif arg == 'BeliefTwinstudy':
            T = TParam
            plotSimulationStudy(arg, priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath, None, adoptionType,
                                endOfExposure, plotVar, startingEnv)

        else:
            T = TParam
            if performSimulation:
                performSimulationAnalysis(arg, priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath, numAgents,
                                          resultsPath, baselineFitness, argumentR, argumentP, None, adoptionType,
                                          False, startingEnv)
            plotSimulationStudy(arg, priorE0Arr, cueValidityC0E0Arr, T, twinResultsPath, None, adoptionType,
                                False, plotVar, startingEnv)
