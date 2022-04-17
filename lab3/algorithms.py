import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import task22


def getEnergy(x, W):
    return -1 * x @ W @ x.T


def getRecallAccuracy(xActual, xReturn):
    # print('xActual: ',xActual)
    l = xActual.shape[0]
    # print(l)
    score = 0
    for i in range(l):
        if xReturn[i] == xActual[i]:
            score += 1
    return score / l


def getPerfectRecall(xActual, xReturn):
    return 1 if np.array_equal(xReturn, xActual) else 0


def getListWiseRecallAccuracyAndPerfectRecallRate(xActuals, xReturns):
    recallAccuracyRate = 0
    perfectRecallRate = 0
    for i, _ in enumerate(xReturns):
        recallAccuracyRate += getRecallAccuracy(xActuals[i], xReturns[i])
        perfectRecallRate += getPerfectRecall(xActuals[i], xReturns[i])
    return recallAccuracyRate / len(xActuals), perfectRecallRate / len(xActuals)


def synchronousUpdate(x, W, sparse=False, theta=0):
    #xOld = None
    xOld = np.copy(x)
    xNew = np.copy(x)
    i = 0
    while True:
        val = W @ xNew
        #val = W @ xOld
        if sparse: 
            xNew = 0.5 + 0.5* np.sign(val-theta)
            #for k in range(xOld.shape[0]):
            #    update_rule = (xOld @ W[k]-theta)        
            #    xNew[k] = 0.5 + 0.5 * np.where(update_rule >= 0, 1, -1)
        else:
            xNew = np.where(val >= 0, 1, -1)

        if np.array_equal(xOld, xNew):
            #print('reaching equal break condition')
            break
        if i > 10000:
            #print('returning with i')
            return np.copy(xNew), i, -1 * xNew @ W @ xNew.T, False
        i += 1
        xOld = np.copy(xNew)
    return np.copy(xNew), i, -1 * xNew @ W @ xNew.T, True


def asynchronousUpdate(x, W, verbose=False, fileName=None, totallyRandomUpdate=False, plot=False):
    xOld = None
    xNew = np.asarray(x)
    j = 1
    k = 0
    randomIndices = np.arange(len(xNew))
    if verbose:
        print("start energy ", -1 * xNew @ W @ xNew.T)

    if fileName:
        its = []
        updates = []
        energies = []

    while True:
        if np.array_equal(xOld, xNew):
            break
        if totallyRandomUpdate:
            updateIndices = np.random.choice(randomIndices,randomIndices.size,replace=True)
        else:
            np.random.shuffle(randomIndices)
            updateIndices = np.copy(randomIndices)
        # print(randomIndices)
        xOld = xNew.copy()
        # print('xnew len: ',len(xNew))
        for i in range(len(xNew)):
            #print('np.sign(W[randomIndices[i], :].shape) : ',W[randomIndices[i], :].shape)
            #print('xNew.shape) : ',xNew.shape)
            val = np.sign(W[updateIndices[i], :] @ xNew)
            # val = np.sign(np.sum)
            # print('this is the val: ', val)
            xNew[updateIndices[i]] = 1 if val >= 0 else -1
            if fileName:
                its.append(j)
                updates.append(j * (i + 1))
                energies.append(-1 * xNew @ W @ xNew.T)
            if k%100==0 and plot==True:
                display(xNew, title=f'p11 recalled after {k} iterations', save=True, filename=f"images/3_2/point3/p11recalled_tot_random_it{k}.png")
            k+=1
        if verbose:
            print("iteration ", k, ", energy: ", -1 * xNew @ W @ xNew.T)

        j += 1
    # print('xold: ',xOld)
    # print('xNEW: ',xNew)
    if fileName:
        df = pd.DataFrame(
            list(zip(its, updates, energies)),
            columns=["num_iterations", "updates", "energy"],
        )
        adress = "graphData/3_3/" + fileName+'.csv'
        df.to_csv(adress)
    return xNew, k, -1 * xNew @ W @ xNew.T


def inNestedList(nested, item):
    """
    Determines if an item is in my_list, even if nested in a lower-level list.
    """
    for l in nested:
        if np.array_equal(l, item):
            return True
    return False


def findAllAttractors(W, len):
    # dataList = list(itertools.product([-1, 1], repeat=len))
    # dataList = [list(i) for i in dataList]
    all_patterns = np.array(
        [np.array(i) for i in itertools.product([-1, 1], repeat=len)]
    )
    # print('dataList: ',len(dataList))
    print("all_patterns: ", all_patterns.shape[0])
    energies = []
    attractors = []
    patterns = []
    for d in all_patterns:
        # d = np.asarray(d)
        # pattern, _, e, _ = synchronousUpdate(d, W)
        pattern, _, e = asynchronousUpdate(d, W)
        # attractors.append(np.asarray(pattern))
        if not inNestedList(attractors, pattern):  # pattern not in attractors:
            energies.append(e)
            attractors.append(np.asarray(pattern))

    attractors = np.asarray(attractors)
    # print(f'attractors: ',attractors.shape[0])
    # attractors = np.unique(attractors, axis=0)
    # print(attractors.shape[0])
    # print(energies)
    # energies = np.array(energies)
    # minEIndices = np.asarray(np.where(energies == energies.min()))
    # print(minEIndices)
    # print('no occurences: ', np.count_nonzero(energies==-24.0))
    # print('no combinations: ', energies.shape)
    # np.argmin(energies, axis=0)
    # print('mine indices', minEIndices)
    # minE = np.ndarray.min(energies)
    # mins = np.where()
    # print('attractors: ', attractors)
    for i, pattern in enumerate(attractors):
        print(f"Pattern {pattern} has energy: {energies[i]}")
    return attractors.shape[0]


def display(image, title="", save=False, filename=""):
    # For task 3.2
    # display images in shape (32, 32) and rotate them so face is up
    plt.figure()
    plt.imshow(np.rot90(image.reshape(32, 32)), origin="lower", interpolation="nearest")
    if title != "":
        plt.title(title)
    if save:
        plt.imsave(filename, (np.rot90(image.reshape(32, 32))))
    plt.show()


def asynchronousUpdatePlot(x, W):
    # print('this is x:',x)
    # print(x.shape)
    xOld = None
    xNew = np.asarray(x)
    j = 1
    randomIndices = np.arange(len(xNew))
    while True:
        if np.array_equal(xOld, xNew):
            break
        np.random.shuffle(randomIndices)

        xOld = np.copy(xNew)

        for i in range(len(xNew)):
            # print(i)
            xNew[randomIndices[i]] = np.sign(W[randomIndices[i], :] @ xNew)
        # if j > 1 and j % 10 == 0:
        #    display(xNew)
        display(xNew)
        j += 1
        # print('xOld',xOld)
        # print('xNew',xNew)
    return xNew, j, -1 * xNew @ W @ xNew.T

