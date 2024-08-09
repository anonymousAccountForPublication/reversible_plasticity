"""
# implementing a forward pass through the optimal policy in order to calculate the
# state distribution matrix
# it indicates the probability of reaching this particular state, following the optimal policy
"""
import numpy as np
import pickle
import time
import os


def set_global_P(setP):
    global P
    P = setP



def set_global_Policy(setcurrPstar):
    global currPstar
    currPstar = setcurrPstar



def compressDict(originalDict):
    newDict = {}
    for key in originalDict:
        a = originalDict[key][1]
        b = originalDict[key][2]
        newDict[key] = (a, b)
    t = {}
    t.update(newDict)
    del newDict
    del originalDict

    return t


def calcForwardPassFinal(currState, survivalProb):
    decisionDict = {0: 1, 1: 3, 2: 4, 3: 5, 4: 6}
    batchResult = {}

    componentList = [int(elem) for elem in currState.split(';')]
    x0, x1, y0BC, y0AC, y1BC, y1AC, y0D, y1D, yw, D0_curr, D1_curr, t = componentList
    (optDecision, pC1_D) = currPstar[currState]
    pC1_D = float(pC1_D)
    nextT = t + 1

    if currState in P:
        currentProb = P[currState]

        if isinstance(optDecision, tuple):
            currentLen = len(optDecision)
            for idx in optDecision:
                phenotype = [y0BC, y0AC, y1BC, y1AC, y0D, y1D, yw, D0_curr, D1_curr]
                phenotype[decisionDict[idx]] += 1
                y0BCNew, y0ACNew, y1BCNew, y1ACNew, y0DNew, y1DNew, ywNew, D0_currNew, D1_currNew = phenotype
                if idx == 2:  # deconstruct 0
                    D0_currNew = nextT
                    y0BCNew = y0BC + y0AC
                    y0ACNew = 0
                if idx == 3:  # deconstruct 1
                    D1_currNew = nextT
                    y1BCNew = y1BC + y1AC
                    y1ACNew = 0

                # now tranform back to strings
                x0add1Key = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (
                x0, x1, y0BCNew, y0ACNew, y1BCNew, y1ACNew, y0DNew, y1DNew, ywNew, D0_currNew, D1_currNew, nextT)

                if not (x0add1Key in batchResult):
                    batchResult[x0add1Key] = 0

                batchResult[x0add1Key] += currentProb * (1 / float(
                    currentLen)) * survivalProb

        else:

            # enumerate all possible successor phenotype states following the optimal policy
            phenotype = [y0BC, y0AC, y1BC, y1AC, y0D, y1D, yw, D0_curr, D1_curr]
            phenotype[decisionDict[optDecision]] += 1
            y0BCNew, y0ACNew, y1BCNew, y1ACNew, y0DNew, y1DNew, ywNew, D0_currNew, D1_currNew = phenotype
            if optDecision == 2:  # deconstruct 0
                D0_currNew = nextT
                y0BCNew = y0BC + y0AC
                y0ACNew = 0
            if optDecision == 3:  # deconstruct 1
                D1_currNew = nextT
                y1BCNew = y1BC + y1AC
                y1ACNew = 0

            # now tranform back to strings
            x0add1Key = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (
                x0, x1, y0BCNew, y0ACNew, y1BCNew, y1ACNew, y0DNew, y1DNew, ywNew, D0_currNew, D1_currNew, nextT)

            if not (x0add1Key in batchResult):
                batchResult[x0add1Key] = 0
            batchResult[x0add1Key] += currentProb * survivalProb

    return batchResult


def calcForwardPass(currState, survivalProb):
    # currKey = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1, y0BC, y0AC, y1BC, y1AC, y0D, y1D, yw, D0_curr,D1_curr,T)

    decisionDict = {0: 1, 1: 3, 2: 4, 3: 5, 4: 6}
    batchResult = {}

    componentList = [int(elem) for elem in currState.split(';')]
    x0, x1, y0BC, y0AC, y1BC, y1AC, y0D, y1D, yw, D0_curr, D1_curr, t = componentList
    (optDecision, pC1_D) = currPstar[currState]
    pC1_D = float(pC1_D)
    nextT = t + 1
    pC0_D = 1 - pC1_D


    if currState in P:
        currentProb = P[currState]

        x0add1 = x0 + 1
        x1add1 = x1 + 1

        if isinstance(optDecision, tuple):
            currentLen = len(optDecision)
            for idx in optDecision:
                phenotype = [y0BC, y0AC, y1BC, y1AC, y0D, y1D, yw, D0_curr, D1_curr]
                phenotype[decisionDict[idx]] += 1
                y0BCNew, y0ACNew, y1BCNew, y1ACNew, y0DNew, y1DNew, ywNew, D0_currNew, D1_currNew = phenotype
                if idx == 2:  # deconstruct 0

                    D0_currNew = nextT
                    y0BCNew = y0BC + y0AC
                    y0ACNew = 0
                if idx == 3:  # deconstruct 1
                    D1_currNew = nextT
                    y1BCNew = y1BC + y1AC
                    y1ACNew = 0

                # now tranform back to strings
                x0add1Key = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (
                x0add1, x1, y0BCNew, y0ACNew, y1BCNew, y1ACNew, y0DNew, y1DNew, ywNew, D0_currNew, D1_currNew, nextT)
                x1add1Key = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (
                x0, x1add1, y0BCNew, y0ACNew, y1BCNew, y1ACNew, y0DNew, y1DNew, ywNew, D0_currNew, D1_currNew, nextT)

                if not (x0add1Key in batchResult):
                    batchResult[x0add1Key] = 0
                if not (x1add1Key in batchResult):
                    batchResult[x1add1Key] = 0

                batchResult[x0add1Key] += currentProb * (1 / float(
                    currentLen)) * pC0_D * survivalProb
                batchResult[x1add1Key] += currentProb * (1 / float(currentLen)) * pC1_D * survivalProb

        else:

            # enumerate all possible successor phenotype states following the optimal policy
            phenotype = [y0BC, y0AC, y1BC, y1AC, y0D, y1D, yw, D0_curr, D1_curr]
            phenotype[decisionDict[optDecision]] += 1
            y0BCNew, y0ACNew, y1BCNew, y1ACNew, y0DNew, y1DNew, ywNew, D0_currNew, D1_currNew = phenotype
            if optDecision == 2:  # deconstruct 0
                D0_currNew = nextT
                y0BCNew = y0BC + y0AC
                y0ACNew = 0
            if optDecision == 3:  # deconstruct 1
                D1_currNew = nextT
                y1BCNew = y1BC + y1AC
                y1ACNew = 0

            # now tranform back to strings
            x0add1Key = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (
                x0add1, x1, y0BCNew, y0ACNew, y1BCNew, y1ACNew, y0DNew, y1DNew, ywNew, D0_currNew, D1_currNew, nextT)
            x1add1Key = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (
                x0, x1add1, y0BCNew, y0ACNew, y1BCNew, y1ACNew, y0DNew, y1DNew, ywNew, D0_currNew, D1_currNew, nextT)

            if not (x0add1Key in batchResult):
                batchResult[x0add1Key] = 0
            if not (x1add1Key in batchResult):
                batchResult[x1add1Key] = 0
            batchResult[x0add1Key] += currentProb * pC0_D * survivalProb
            batchResult[x1add1Key] += currentProb * pC1_D * survivalProb

    return batchResult



"""
This is the main function that is called 

    steps
    - make a plotting folder; check 
    - iterate over time and have one folder per time step
    - work in batches, i.e. define a batchsize parameter from the main script
    - always need t and t+1 in memory 
"""


def doForwardPass(T, finalTimeArr):
    # start the clock
    startTime = time.perf_counter()
    # the current working directory is the folder for the respective parameter combination
    # a plotting folder exists
    # add a state distribution folder


    stateDistPath = 'plotting/StateDistribution'

    if not os.path.exists(stateDistPath):
        os.makedirs(stateDistPath)

    tValues = np.arange(1, T + 1, 1)  # This needs to run from 1 ? till 19!

    """
    This is the initialization 
    - here we basically initialize a state distribution vector for t equals 0 and determine how the population starts 
    """
    print("start initialization")
    # make one folder per time step
    if not os.path.exists(os.path.join(stateDistPath, str(0))):
        os.makedirs(os.path.join(stateDistPath, str(0)))

    localP = {}
    # this is the initialization
    currPstarLocal = pickle.load(open("fitness/0/TF.p", 'rb'))

    for currKey in currPstarLocal:
        localP[currKey] = 0
    currKey = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    localP[currKey] = 1  # this where the whole population starts

    tree0 = pickle.load(open("trees/tree0.p", 'rb'))

    pC0D, pC1D = tree0[0, 0][1]  # just need to do this for the initialization

    initialSurvProb = 1 - float(finalTimeArr[1])  # TODO think more carefully about this bit; needs to be one longer
    # currKey = '%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (x0, x1, y0BC, y0AC, y1BC, y1AC, y0D, y1D, yw, D0_curr,D1_curr,T)
    localP['%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)] = pC1D * localP[
        currKey] * initialSurvProb
    localP['%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s' % (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)] = pC0D * localP[
        currKey] * initialSurvProb

    # dump it in the respective folder
    pickle.dump(localP, open(os.path.join(stateDistPath, "%s/P.p" % 0),
                             "wb"))  # I know what the distribution is for each state in time step 1

    del currPstarLocal
    print('finished initialization')

    """
    Here, the actual calculation of the state distribution matrix will begin 
    - this will constitute a forward pass  
    - the resulting P matrices are a one-to-one mapping from policy states, if a state from the policy is not in the 
        in the P matrix, it's state distribution can be assumed to be zero 
    """

    # initialize a global dicts for P and policy star so that all parallel workers have access to it

    for t in tValues:
        print("currently computing the forward pass for time step %s" % t)
        global currPstar
        currPstar = {}
        global P
        P = {}
        currentP = {}

        if not os.path.exists(os.path.join(stateDistPath, str(t))):
            os.makedirs(os.path.join(stateDistPath, str(t)))

        set_global_Policy(compressDict(pickle.load(open("fitness/%s/TF.p" % (t - 1), 'rb'))))

        if os.path.exists(os.path.join(stateDistPath, "%s/P.p" % (t - 1))):
            set_global_P(pickle.load(open(os.path.join(stateDistPath, "%s/P.p" % (t - 1)), 'rb')))
        else:
            print("No such file")



        # next call the forwardPass function
        survivalProb = 1 - float(finalTimeArr[t])

        parallelStates = currPstar.keys()

        if t < T:
            results = [calcForwardPass(currState, survivalProb) for currState in parallelStates]
        else:
            results = [calcForwardPassFinal(currState, survivalProb) for currState in parallelStates]

        for tempResult in results:
            for state in tempResult.keys():
                if state in currentP:
                    currentP[state] += tempResult[state]
                else:
                    currentP[state] = tempResult[state]

        del results

        """
        store the results
        
        """

        pickle.dump(currentP, open(os.path.join(stateDistPath, "%s/P.p" % t), 'wb'))

        #del P

    print("Elapsed time for thr forward pass: " + str(time.perf_counter() - startTime))
    

