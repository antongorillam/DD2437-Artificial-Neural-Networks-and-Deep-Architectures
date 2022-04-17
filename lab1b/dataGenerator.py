"""
Store all the data generating function for each task here
"""
from importlib import import_module
from matplotlib.pyplot import figure
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def getData(seed, N=100, mA=[1.0, 0.3], mB=[0.0,-0.1], sigmaA=0.2, sigmaB=0.3):
    
    classA = np.zeros((2, N))  # init
    classB = np.zeros((2, N))  # init
    np.random.seed(seed)

    # print(f"concat: {}")
    # change classA[0]
    classA[0] = np.concatenate(
        (
            np.random.normal(-mA[0], sigmaA, round(0.5 * N)),
            np.random.normal(+mA[0], sigmaA, round(0.5 * N)),
        )
    )
    classA[1] = np.random.normal(mA[1], sigmaA, N)
    classB[0] = np.random.normal(mB[0], sigmaB, N)
    classB[1] = np.random.normal(mB[1], sigmaB, N)

    return classA, classB

def dataSample1(classA, classB, seed):
    """ Random 25% from each class """
    np.random.seed(seed)
    np.random.shuffle(classB)
    # classB npng.random.shuffle(classB)
    
    numRowsA = classA.shape[1]
    numRowsB = classB.shape[1]
    onesCol = np.ones((1, numRowsB))
    negOnesCol = -1 * np.ones((1, numRowsA))
    
    labeledA = np.append(classA, negOnesCol, axis=0)
    labeledB = np.append(classB, onesCol, axis=0)

    classATrain, classAVal, classBTrain, classBVal = train_test_split(labeledA.T, labeledB.T, test_size=25, random_state=seed)
    return classATrain.T, classAVal.T, classBTrain.T, classBVal.T

def dataSample2(classA, classB, seed):
    """ Random 50% from classA """
    np.random.seed(seed)
    np.random.shuffle(classB)

    numRowsA = classA.shape[1]
    numRowsB = classB.shape[1]
    onesCol = np.ones((1, numRowsB))
    negOnesCol = -1 * np.ones((1, numRowsA))

    labeledA = np.append(classA, negOnesCol, axis=0)
    labeledB = np.append(classB, onesCol, axis=0)

    classATrain, classAVal = train_test_split(labeledA.T, test_size=0.5, random_state=seed)
    return classATrain.T, classAVal.T, labeledB 

def dataSample3(classA, classB, seed):
    """ 20% from a subset of classA for which classA(1,:)<0 and 80% from a subset of classA for which classA(1,:)>0 """

    numRowsB = classB.shape[1]

    onesCol = np.ones((1, numRowsB))
    # negOnesCol = -1 * np.ones((1, numRowsA))
    labeledB = np.append(classB, onesCol, axis=0)
    
    classANegative = np.array([elem for elem in classA.T if elem[1] < 0]).T
    classAPositive = np.array([elem for elem in classA.T if elem[1] > 0]).T
    numRowsNegA = classANegative.shape[1]
    numRowsPosA = classAPositive.shape[1]
    negOnesCol1 = -1 * np.ones((1, numRowsNegA))
    negOnesCol2 = -1 * np.ones((1, numRowsPosA))
    
    negLabeledA = np.append(classANegative, negOnesCol1, axis=0)
    posLabeledA = np.append(classAPositive, negOnesCol2, axis=0)

    classATrainNegative, classAValNegative = train_test_split(negLabeledA.T, test_size=0.2,  random_state=seed)
    classATrainPositive, classAValPositive = train_test_split(posLabeledA.T, test_size=0.8, random_state=seed)

    classATrain = np.concatenate((classATrainNegative, classATrainPositive))
    classAVal = np.concatenate((classAValNegative, classAValPositive))

    return classATrain.T, classAVal.T, labeledB


def addLabels(classA, classB, seed):
    """ Add labels and concatinates the classas
    classA: extra column with  -1:s will be added
    classB: extra column with 1:s will be added
    
    return: labeled
    """
    np.random.seed(seed)

    numRowsA = classA.shape[1]
    numRowsB = classB.shape[1]
    onesCol = np.ones((1, numRowsB))
    negOnesCol = -1 * np.ones((1, numRowsA))

    labeledA = np.append(classA, negOnesCol, axis=0)
    labeledB = np.append(classB, onesCol, axis=0)
    
    labeledAB = np.append(labeledA, labeledB, axis=1).T
    np.random.shuffle(labeledAB)  # Shuffles the classes
    target = labeledAB[:, 2:3]
    x_input = labeledAB[:, :2]
    return x_input.T, target.T

def x(tMax, beta=0.2, gamma=0.1, n=10, tau=25):
    xs = np.zeros(tMax+50)
    xs[1] = 1.5
    for t in range(2, tMax):    
        if t-tau-1 < 0:
            xs[t] = xs[t-1] + 0 - (gamma * xs[t-1])
        else:            
            xs[t] = xs[t-1] + (beta * xs[t-tau-1] / (1 + xs[t-tau-1]**n)) - (gamma * xs[t-1])
    
    return xs

def getMakeyGlassData(trainSize=0.5, plot=False):
    tMin = 300
    tMax = 1500
    xs = x(tMax)

    input = np.array([
        xs[tMin-20 : tMax-20],
        xs[tMin-15 : tMax-15],
        xs[tMin-10 : tMax-10],
        xs[tMin-5 : tMax-5],
        xs[tMin : tMax],
    ])

    output = xs[tMin+5 : tMax+5]

    if plot:
        filename = "Mackey-Glass Time-serie"
        fig = plt.figure()
        xMin = 301
        xMax = 1505
        ts = np.arange(xMin, xMax)
        plt.title(filename)
        plt.xlabel("t")
        plt.plot(ts, xs[xMin:xMax])
        plt.grid()
        filename = filename.replace(" ", "_")
        filename = filename.replace(".", "")
        plt.savefig('images/4.3.1_time_serie/' + filename)

    inputTestData = input[:,-200:]
    outputTestData = output[-200:]
    inputTrainValData = input[:,:-200]
    outputTrainValData = output[:-200]
    trainNum = int(inputTrainValData.shape[1]*trainSize)
    inputTrainData = inputTrainValData[:,:trainNum]
    outputTrainData = outputTrainValData[:trainNum]
    inputValidationData = inputTrainValData[:,trainNum:]
    outputValidationData = outputTrainValData[trainNum:]

    # output = {
    #     "inputTrainData": inputTrainData,
    #     "outputTrainData": outputTrainData,
    #     "inputValidationData": inputValidationData,
    #     "outputValidationData": outputValidationData,
    #     "inputTestData": inputTestData,
    #     "outputTestData": outputTestData,
    # }

    return inputTrainData, outputTrainData, inputValidationData, outputValidationData, inputTestData, outputTestData

def getDataWithNoise(std):
    inputTrainData, outputTrainData, inputValidationData, outputValidationData, inputTestData, outputTestData = getMakeyGlassData(plot=False, trainSize=0.7)
    print('input train shape: ', inputTrainData.shape)
    inputTrainData = inputTrainData + np.random.normal(scale=std,size=inputTrainData.shape)

    return inputTrainData, outputTrainData, inputValidationData, outputValidationData, inputTestData, outputTestData  


""" For testing purposes """
if __name__ == "__main__":
    
    input = getMakeyGlassData(plot=True)
    
    # fig = plt.figure()
    # plt.plot(mcKay)
    # plt.show()
    # ts = np.arange(301, 1500+1)
      
    # print(x(ts))

    
'''
noSamples = array.shape[1]
    keepSize = int(noSamples * remainPart)
    valSize = int(noSamples * (1 - remainPart))
    keepIndices = np.sort(
        np.random.choice(range(noSamples), size=keepSize, replace=False)
    )
    
    allIndices = [i for i in range(noSamples)]
    print('allindices', allIndices)
    # double check that valIndices is correctly implemented
    valIndices = np.sort([i for i in allIndices if i not in keepIndices])
    subArr = np.zeros((2, keepSize))
    valArr = np.zeros((2, valSize))
    print('keep: ', keepIndices, keepIndices.shape)
    print('val: ', valIndices, valIndices.shape)

    for i in range(keepSize):
        subArr[0, i] = array[0, keepIndices[i]]
        subArr[1, i] = array[1, keepIndices[i]]

    #for j in range(valSize):
    #    valArr[0,i] = array[0, valIndices[i]]
    #    valArr[1,i] = array[1, valIndices[i]]

    subArr, valArr, _, _ = train_test_split(classA, _, test_size=1-remainPart,seed=SEED)
'''