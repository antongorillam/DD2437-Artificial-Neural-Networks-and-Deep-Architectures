from re import X

import matplotlib.pyplot as plt
import numpy as np

<<<<<<< HEAD
=======

>>>>>>> refs/remotes/origin/main
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

<<<<<<< HEAD
def getRandomPatterns(patterns, neurons, bias = False):
    storedPatterns = []
    for i in range(0, patterns):
        pRandom = np.random.randint(0, 2, (1, neurons))
        
        storedPatterns.append(pRandom[0])
=======
def getActivePatterns(patterns, neurons, active_patterns):
    storedPatterns = []
    for i in range(0, patterns):
        pattern = np.zeros((1,neurons))
        n = int(active_patterns * pattern.shape[1])
        index = np.random.choice(pattern.shape[1], n,)
        for i in index:
            pattern[0, i] = 1 
        storedPatterns.append(pattern[0])
>>>>>>> refs/remotes/origin/main
    return np.asarray(storedPatterns)

def getDistortion(x, part):
    x = x.reshape((1, x.shape[0]))
    n = int(part * x.shape[1])
    index = np.random.choice(x.shape[1], n, replace=False)
    for i in index:
        x[0, i] = 1 if x[0, i] == 0 else 1
    return np.copy(x)

<<<<<<< HEAD

if __name__ == "__main__":
    activity = 0.01
    thetas = []
    patterns = getRandomPatterns(20, 100)
    pattern1 = patterns[0,:]
    W = dataGenerator.getW36(patterns=[pattern1], p=activity)
    pattern1 = patterns[0,:]
    pattern1Dist = getDistortion(pattern1, 0.0)
    print(f'pattern1Dist: {pattern1Dist}')
    print(f'equal?: {np.array_equal(pattern1, pattern1Dist)}')
    xOut,i,_,_ = algorithms.synchronousUpdate(pattern1Dist[0], W, sparse=True, theta=2)
    stable = algorithms.getPerfectRecall(pattern1, xOut)
    print(f'equal?: {np.array_equal(xOut, pattern1Dist)}')
    accuracy = algorithms.getRecallAccuracy(pattern1, xOut)
    print(f'pattern1 accuracy: {accuracy}, Stable Patterns: {stable}, with {i} iterations')
=======
def point1(patterns, thetas, activity):
    
    results = []
    for i, theta in enumerate(thetas):
        maxStability = 0
        for j in range(1,patterns.shape[0]):
            stability = 0
            storedPatterns = patterns[:j,:]
            W = dataGenerator.getW36(storedPatterns, p=activity)
            for k, pattern in enumerate(storedPatterns):
                xOut,_,_,_ = algorithms.synchronousUpdate(pattern, W, sparse=True, theta=theta)
                stability += algorithms.getPerfectRecall(pattern, xOut)
            if stability == storedPatterns.shape[0]:
                
                maxStability = stability
        results.append([theta, maxStability])
    return results



def plotActivity(input):
    plt.figure()
    
    
    
    for i, data in enumerate(input):
        activity = data[0]
        result = data[1]
        plt.plot([res[0] for res in result], [res[1] for res in result],label=f'{str(activity*100)} % active patterns')
    plt.title('Number of stored patterns per activity and theta')   
    plt.xlabel("Theta")
    plt.ylabel("Number of stored stable patterns")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    activities = [0.1, 0.025,0.05,0.075,0.01]
    #activity = 0.1
    thetas = np.arange(1,10,0.25, dtype=float)
    fullResults = []
    #print('thetas: ', thetas)
    for activity in activities:
        patterns = getActivePatterns(20, 100, activity)
    
        results = point1(patterns, thetas, activity)
        fullResults.append([activity,results])

    plotActivity(fullResults)
    
    
    # pattern1 = patterns[0,:]
    # W = dataGenerator.getW36(patterns=[pattern1], p=activity, scaling=False)
    # pattern1 = patterns[0,:]
    # pattern1Dist = getDistortion(pattern1, 0.0)
    # print(f'pattern1Dist: {pattern1Dist}')
    # unique, counts = np.unique(pattern1Dist[0], return_counts=True)
    # print(dict(zip(unique, counts)))
    # print(f'equal?: {np.array_equal(pattern1, pattern1Dist)}')
    # xOut,i,_,_ = algorithms.synchronousUpdate(pattern1Dist[0], W, sparse=True, theta=2)
    # stable = algorithms.getPerfectRecall(pattern1, xOut)
    # print(f'equal?: {np.array_equal(xOut, pattern1Dist)}')
    # accuracy = algorithms.getRecallAccuracy(pattern1, xOut)
    # print(f'pattern1 accuracy: {accuracy}, Stable Patterns: {stable}, with {i} iterations')
>>>>>>> refs/remotes/origin/main

    

