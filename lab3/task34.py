from re import X

import matplotlib.pyplot as plt
import numpy as np

import algorithms
import dataGenerator
import task22
import pandas as pd
# n=500
# x= None
patterns = dataGenerator.getPatterns(directory="data/pict.dat", neurons=1024)
first3_patterns = patterns[:3, :]
W = dataGenerator.getW(first3_patterns)
p1 = patterns[0, :]
p2 = patterns[1, :]
p3 = patterns[2, :]
p4 = patterns[3, :]
p5 = patterns[4, :]
p6 = patterns[5, :]
p7 = patterns[6, :]
p8 = patterns[7, :]
p9 = patterns[8, :]
p10 = patterns[9, :]
p11 = patterns[10, :]
spuriousp1 = np.negative(p1)
spuriousp2 = np.negative(p2)
spuriousp3 = np.negative(p3)

def getDistortion(x, part):
    x = x.reshape((1, 1024))
    n = int(part * x.shape[1])
    index = np.random.choice(x.shape[1], n, replace=False)
    for i in index:
        x[0, i] = 1 if x[0, i] == -1 else 1
    return np.copy(x)


def checkRestoration(xOrigin, xDistorted,pattern):
    # kalla på algoritmen
    #print(xOrigin)
    #print(len(xDistorted))
    xOut, _, _,_ = algorithms.synchronousUpdate(x=xDistorted, W=W)
    #print(xOrigin.shape)
    #print(xOut.shape)

    allSame = True
    allSameDist = True
    for i,val in enumerate(xOrigin):
        if xOrigin[i] != xOut[i]:
            allSame = False
        if xOrigin[i] != xDistorted[i]:
            allSameDist = False
    #print('value of allsame xorigin vs xout: ', allSame)
    #print('value of allsame xorigin vs xdist: ', allSameDist)
    if np.array_equal(xOrigin, xOut):
        print('converged to same attractor pattern')
        return True
    else:
        printBool = False
        for i,name in enumerate(patternNames):
            if pattern == name and np.array_equal(xOut,spuriousPatterns[i]):
                print('converged to same spurious pattern')
                printBool = True
        if not printBool:
            for pattern in patterns:
                if np.array_equal(xOut,pattern):
                    printBool = True
                    print('converged to other pattern')
        if not printBool:
            for pattern in spuriousPatterns:
                if np.array_equal(xOut,pattern):
                    print('converged to other purious pattern')
                    printBool = True
        if not printBool:
            print('did not converge to any above')
        return False
    




if __name__ == "__main__":
    patterns = [p1, p2, p3]
    spuriousPatterns = [spuriousp1, spuriousp2, spuriousp3]
    print(p1)
    patternsList = []
    patternNames = ["p1", "p2", "p3"]
    parts = [0.0, 0.05, 0.1, 0.15, 0.2,0.25, 0.3,0.35, 0.4,0.45, 0.5,0.55, 0.6,0.65, 0.7,0.75, 0.8,0.85, 0.9,0.95, 1.0]
    
    for i, pattern in enumerate(patterns):
        for part in parts:
            trueCount = 0
            for j in range(100):
                
                xDist = getDistortion(np.copy(pattern), part)
            #print('pattern: ', pattern)
            #print('xDist: ', xDist)
                val = checkRestoration(pattern, xDist[0],patternNames[i])
                if val == True:
                    trueCount += 1
                #print(patternNames[i], " distortion part: ", part, " outcome: ", val)
            
            patternsList.append([patternNames[i],part,trueCount])

    df = pd.DataFrame(patternsList,columns=['Pattern', 'distortion','score'])
    adress = 'graphData/34data.csv'
    df.to_csv(adress)
    

