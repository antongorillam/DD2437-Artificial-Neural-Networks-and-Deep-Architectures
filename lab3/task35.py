import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import algorithms
import dataGenerator

# np.random.seed(50)

patterns = dataGenerator.getPatterns(directory="data/pict.dat", neurons=1024)


def checkRestoration(xOrigin, xDistorted, W):
    # kalla på algoritmen
    print(xDistorted)
    xOut, _, _, _ = algorithms.synchronousUpdate(xDistorted, W)
    algorithms.display(xOrigin)
    algorithms.display(xOut)

    return algorithms.getRecallScore(xOut, xOrigin)


def getDistortion(x, part):
    x = x.reshape((1, x.shape[0]))
    n = int(part * x.shape[1])
    index = np.random.choice(x.shape[1], n, replace=False)
    for i in index:
        x[0, i] = 1 if x[0, i] == -1 else 1
    return np.copy(x)


def getBias(x, part):
    print(x.shape)
    n = int(part * x.shape[0])
    index = np.random.choice(x.shape[1], n, replace=False)
    for i in index:
        if x[0, i] == -1:
            x[0, i] = 1
    return np.copy(x)


def point1(patterns):

    p1 = patterns[0, :]
    p2 = patterns[1, :]
    p3 = patterns[2, :]
    p4 = patterns[3, :]
    p5 = patterns[4, :]
    p6 = patterns[5, :]
    p7 = patterns[6, :]
    first_3_patterns = patterns[:3, :]
    first_4_patterns = patterns[:4, :]
    
    

    #Learn Pattern 1-3
    print(f'Learn Pattern 1-3')
    pattern_to_store = first_3_patterns
    W = dataGenerator.getW(pattern_to_store)
    print(f'Attempt to recall p1 from p1')
    equal = checkRestoration(p1,p1)
    print(f'with {equal} with pattern 1-3 stored')

    print(f'Attempt to recall p2 from p2')
    equal = checkRestoration(p2,p2)
    print(f'with {equal} with pattern 1-3 stored')

    print(f'Attempt to recall p3 from p3')
    equal = checkRestoration(p3,p3)
    print(f'with {equal} with pattern 1-3 stored')
    
    #Learn Pattern 1-4
    print(f'Learn Pattern 1-4')
    pattern_to_store = first_4_patterns
    W = dataGenerator.getW(pattern_to_store)
    print(f'Attempt to recall p4 from p4')
    equal = checkRestoration(p4,p4)
    print(f'with {equal} with pattern 1-4 stored')

    # Learn Pattern 5-7
    print(f"Learn Pattern 1, 5-7")
    pattern_to_store = patterns[5:7, :]
    W = dataGenerator.getW(np.append(pattern_to_store, [p2], axis=0))

    print(f"Attempt to recall p5 from p5")
    equal = checkRestoration(p5, p5)
    print(f"with recall score {equal} with pattern 1-7 except 4 stored")

    print(f"Attempt to recall p6 from p6")
    equal = checkRestoration(p6, p6)
    print(f"with recall score {equal} with pattern 1-7 except 4 stored")

    print(f"Attempt to recall p7 from p7")
    equal = checkRestoration(p7, p7)
    print(f"with recall score {equal} with pattern 1-7 except 4 stored")



# def point2(numPatterns, neurons):
#     xInXOut =[]
#     # for r in range(1,10)
#     random_patterns = getRandomPatterns(patterns=numPatterns, neurons=neurons)
#     W = dataGenerator.getW(patterns=random_patterns)
#     for i, pattern in enumerate(random_patterns):
#         distortedPattern = getDistortion(pattern, 0.05)
#         xOut,_,_,_ = algorithms.synchronousUpdate(distortedPattern[0], W)
#         xInXOut.append([pattern, xOut])
#     print('xinxout', xInXOut)

#


def point2(numPatterns, neurons):
    xInXOut = []
    for r in range(1, 200):
        random_patterns = getRandomPatterns(patterns=r, neurons=neurons)
        W = dataGenerator.getW(patterns=random_patterns)
        patterns = []
        xOuts = []
        for i, pattern in enumerate(random_patterns):
            distortedPattern = getDistortion(pattern, 0.00)
            xOut, _, _ = algorithms.asynchronousUpdate(distortedPattern[0], W)
            xOuts.append(xOut)
            patterns.append(pattern)
        xInXOut.append([patterns, xOuts])

    graphData = getGraphData(listOfPatterns=xInXOut, noise="0.00")
    get35GraphDF(graphData, noiseLevel="0.00")


def point45():
    randomPatterns = getRandomPatterns(patterns=300, neurons=100)
    print(f"randomPatterns.shape: {randomPatterns.shape}")
    stabilities = []
    noisyStabilities = []
    for i in range(1, randomPatterns.shape[0]):
        accuracy = []
        noisyAccuracy = []
        stability = 0
        noisyStability = 0
        storedPatterns = randomPatterns[:i, :]
        W = dataGenerator.getW(storedPatterns)
        for j, pattern in enumerate(storedPatterns):
            xOut, _, _, _ = algorithms.synchronousUpdate(pattern, W)
            if algorithms.getPerfectRecall(pattern, xOut) == 1:
                stability += 1
            distortedPattern = getDistortion(pattern, 0.01)
            xOutNoisy, _, _, _ = algorithms.synchronousUpdate(distortedPattern[0], W)
            if algorithms.getPerfectRecall(pattern, xOutNoisy) == 1:
                noisyStability += 1
        print(f"Number of stable patterns for {i} stored patterns: {stability}")
        print(
            f"Number of stable patterns for {i} stored patterns with noisy data: {noisyStability}"
        )
        stabilities.append(stability)
        noisyStabilities.append(noisyStability)
    plotCompare(stabilities, noisyStabilities)


def point67():
    randomPatterns = getRandomPatterns(patterns=300, neurons=100, bias=True)
    print(f"randomPatterns.shape: {randomPatterns.shape}")
    stabilities = []
    noisyStabilities = []
    for i in range(1, randomPatterns.shape[0]):
        accuracy = []
        noisyAccuracy = []
        stability = 0
        noisyStability = 0
        storedPatterns = randomPatterns[:i, :]
        print(f"storedPatterns.shape: {storedPatterns.shape}")

        W = dataGenerator.getW(storedPatterns, selfConnect=True, scaling=True)
        # print('W: ', W)
        # print('mean: ',np.mean(W))

        for j, pattern in enumerate(storedPatterns):

            # print('pattern: ', pattern)
            # print('mean: ',np.mean(pattern))
            xOut, _, _, _ = algorithms.synchronousUpdate(pattern, W)
            if algorithms.getPerfectRecall(pattern, xOut) == 1:
                stability += 1
            distortedPattern = getDistortion(pattern, 0.01)
            xOutNoisy, _, _, _ = algorithms.synchronousUpdate(distortedPattern[0], W)
            if algorithms.getPerfectRecall(pattern, xOutNoisy) == 1:
                noisyStability += 1
        print(f"Number of stable patterns for {i} stored patterns: {stability}")
        print(
            f"Number of stable patterns for {i} stored patterns with noisy data: {noisyStability}"
        )
        stabilities.append(stability)
        noisyStabilities.append(noisyStability)
    plotCompare(stabilities, noisyStabilities)


def plotCompare(data1, data2):

    plt.figure()
    plt.plot(data1, color="b", label="Original patterns")
    plt.plot(data2, color="r", label="Noisy patterns")
    plt.ylabel("Number of stable patterns")
    plt.xlabel("Number of stored patterns")
    plt.legend()
    plt.savefig(f"images/stability_no_noise_vs_noise_point7")


def getRandomPatterns(patterns, neurons, bias=False):
    storedPatterns = []
    for i in range(0, patterns):
        pRandom = np.random.randint(0, 2, (1, neurons))
        for i, x in enumerate(pRandom[0]):
            if x < 0.5:
                pRandom[0][i] = -1
        if bias:
            pRandom = getBias(pRandom, 0.5)
        storedPatterns.append(pRandom[0])
    return np.asarray(storedPatterns)


# xOrigins: original data
# xReturns: return data from running hopfield on the distorted data
#     # 5 patterns             # 10 patterns
# [[xIns1: [in1,in2,in3...], xOuts1: [out1,out2,ou3,...]], [xIns2, xOuts2]]
def getGraphData(listOfPatterns, noise):
    graphData = []
    for patternsN in listOfPatterns:
        # print('patternsN: ',patternsN)
        (
            recallAccuracyRate,
            perfectRecallRate,
        ) = algorithms.getListWiseRecallAccuracyAndPerfectRecallRate(
            patternsN[0], patternsN[1]
        )
        graphData.append(
            [len(patternsN[0]), noise, recallAccuracyRate, perfectRecallRate]
        )
    return graphData


def get35GraphDF(graphData, noiseLevel):
    df = pd.DataFrame(
        graphData,
        columns=[
            "Number of patterns",
            "noise level",
            "recall accuracy rate",
            "perfect recall rate",
        ],
    )
    adress = "graphData/3_5/" + noiseLevel + "v2" + ".csv"
    df.to_csv(adress)
    return None


if __name__ == "__main__":
    p1 = patterns[0, :]
    p2 = patterns[1, :]
    p3 = patterns[2, :]
    p1Deg = patterns[9, :]
    p2_3 = patterns[10, :]

    first_3_patterns = patterns[:3, :]
    p4 = patterns[3, :]
    p5 = patterns[4, :]
    p6 = patterns[5, :]
    p7 = patterns[6, :]

    # point1(patterns)

    point2(numPatterns=5, neurons=1024)
    # point45()
    # point67()

    # print(f'display 5')

    # #Learn Pattern 1-3
    # print(f'Learn Pattern 1-3')
    # pattern_to_store = first_3_patterns
    # W = dataGenerator.getW(pattern_to_store)
    # print(f'Attempt to recall p1 from p1')
    # equal = 'success'if checkRestoration(p1,p1) else 'failure'
    # print(f'with {equal} with pattern 1-3 stored')

    # print(f'Attempt to recall p2 from p2')
    # equal = 'success'if checkRestoration(p2,p2) else 'failure'
    # print(f'with {equal} with pattern 1-3 stored')

    # print(f'Attempt to recall p3 from p3')
    # equal = 'success'if checkRestoration(p3,p3) else 'failure'
    # print(f'with {equal} with pattern 1-3 stored')

    # #Learn Pattern 1-4
    # print(f'Learn Pattern 1-4')
    # pattern_to_store = np.append(first_3_patterns,[pattern_4], axis=0)
    # W = dataGenerator.getW(pattern_to_store)
    # print(f'Attempt to recall p4 from p4')
    # equal = 'success'if checkRestoration(pattern_4,pattern_4) else 'failure'
    # print(f'with {equal} with pattern 1-4 stored')

    # #Learn Pattern 1-5
    # print(f'Learn Pattern 1-5')
    # pattern_to_store = np.append(first_3_patterns,[pattern_4, pattern_5], axis=0)
    # W = dataGenerator.getW(pattern_to_store)
    # print(f'Attempt to recall p4 from p4')
    # equal = 'success'if checkRestoration(pattern_4,pattern_4) else 'failure'
    # print(f'with {equal} with pattern 1-5 stored')

    # print(f'Attempt to recall p5 from p5')
    # equal = 'success'if checkRestoration(pattern_5,pattern_5) else 'failure'
    # print(f'with {equal} with pattern 1-5 stored')

    # #Learn Pattern 1-5 except 4
    # print(f'Learn Pattern 1-5 except 4')
    # pattern_to_store = np.append(first_3_patterns,[pattern_5], axis=0)
    # W = dataGenerator.getW(pattern_to_store)
    # print(f'Attempt to recall p5 from p5')
    # equal = 'success'if checkRestoration(pattern_5,pattern_5) else 'failure'
    # print(f'with {equal} with pattern 1-5 except 4 stored')

    # #Learn Pattern 1-6 except 4
    # print(f'Learn Pattern 1-6 except 4')
    # pattern_to_store = np.append(first_3_patterns,[pattern_5, pattern_6], axis=0)
    # W = dataGenerator.getW(pattern_to_store)
    # print(f'Attempt to recall p5 from p5')
    # equal = 'success'if checkRestoration(pattern_5,pattern_5) else 'failure'
    # print(f'with {equal} with pattern 1-6 except 4 stored')

    # print(f'Attempt to recall p6 from p6')
    # equal = 'success'if checkRestoration(pattern_6,pattern_6) else 'failure'
    # print(f'with {equal} with pattern 1-6 except 4 stored')

    # #Learn Pattern 1-7 except 4
    # print(f'Learn Pattern 1-7 except 4')
    # pattern_to_store = np.append(first_3_patterns,[pattern_5, pattern_6, pattern_7], axis=0)
    # W = dataGenerator.getW(pattern_to_store)
    # print(f'Attempt to recall p5 from p5')
    # equal = 'success'if checkRestoration(pattern_5,pattern_5) else 'failure'
    # print(f'with {equal} with pattern 1-7 except 4 stored')

    # print(f'Attempt to recall p6 from p6')
    # equal = 'success'if checkRestoration(pattern_6,pattern_6) else 'failure'
    # print(f'with {equal} with pattern 1-7 except 4 stored')

    # print(f'Attempt to recall p7 from p7')
    # equal = 'success'if checkRestoration(pattern_7,pattern_7) else 'failure'
    # print(f'with {equal} with pattern 1-7 except 4 stored')

    # for i in range(1,patterns.shape[0]):
    #     first_i_patterns = patterns[:i, :]
    #     W = dataGenerator.getW(first_i_patterns)

    #     #Try to recall p1, p2 & p3
    #     print('--- --- ---')
    #     print(f'Attempt at storing {i} patterns')
    #     print('--- --- ---')
    #     print(f'Attempt to recall p1 from p1 {checkRestoration(p1,p1)}')
    #     print(f'Attempt to recall p2 from p2 {checkRestoration(p2,p2)}')
    #     print(f'Attempt to recall p3 from p3 {checkRestoration(p3,p3)}')
    #     #print(f'Attempt to recall p1 from p10 {checkRestoration(p1,p1Deg)}')
    #     #print(f'Attempt to recall p2 from p11 {checkRestoration(p2,p2_3)}')
    #     #print(f'Attempt to recall p3 from p11 {checkRestoration(p3,p2_3)}')
