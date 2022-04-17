from distutils.log import error
from fileinput import filename

import matplotlib.pyplot as plt
import numpy as np

import dataGenerator
from rbf import RBF
from rbf_CL import RBFCL


def getMSE(yPred, yTest):
    mse = sum((yPred - yTest) ** 2) / len(yTest)
    return mse[0]

def plotConvergence(epochs, errors, titles):
    x = [i for i in range(1, epochs)]
    filename = f"Convergence on sin(2x) for RBF with and without competitive learning\n and with and without noisy data"
    plt.figure()
    plt.title(filename)
    for ie, e in enumerate(errors):
        plt.plot(x, e, label=titles[ie])
    plt.xlabel("Number of epochs")
    plt.ylabel("Residual Error")
    plt.legend()
    plt.grid()
    # filename = filename.replace(" ", "_").replace("-", "").replace("\n", "")
    # plt.savefig('images/RBF_CL/point1/'+ filename)

def plotMSE(errors, titles):
    print("errors: ", errors)
    plt.figure()
    plt.title("Test MSE of the different configurations after 50 epochs")
    x = np.arange(len(errors))
    plt.bar(x=x, height=errors, tick_label=titles)
    # for ie, e in enumerate(errors):
    #     plt.bar(x=x[ie], height=e,tick_label=titles[ie])
    plt.show()


def plotEpochMSEs(mses, titles):
    plt.figure()
    plt.title("MSE of the different configurations per epoch")
    x = np.arange(len(mses[0]))
    for i, mse in enumerate(mses):
        plt.plot(x, mse, label=titles[i])
    plt.legend()
    plt.show()
    pass


def plotNodes(rbf, xTrain, label):

    plt.figure()
    titleString = "RBF units placements around input space, " + label
    plt.title(titleString)
    centers = rbf.centers

    plt.scatter(centers[:,0], centers[:,1], c="b", label="RBF centers")
    plt.scatter(xTrain[:,0], xTrain[:,1], c="r", label="xTrain")
    centersSin = [np.sin(2 * i) for i in centers]
    centersSin = [i[0] for i in centersSin]
    # print('centerssin ', centersSin)
    plt.legend()
    plt.grid()
    # filename = titleString.replace(" ", "_").replace("-", "").replace("\n", "")
    # plt.savefig('images/RBF_CL/point1/'+ filename)


# non-noisy data
# sinData = dataGenerator.getSin(noise=False)
# xTrain, yTrain, xTest, yTest = sinData["xTrain"], sinData["yTrain"], sinData["xTest"], sinData["yTest"]

# noisy data
# sinData = dataGenerator.getSin(noise=True)
# xTrain, yTrain, xTest, yTest = sinData["xTrain"], sinData["yTrain"], sinData["xTest"], sinData["yTest"]
# squareData = dataGenerator.getSquare(noise=True)
# xTrain, yTrain, xTest, yTest = squareData["xTrain"], squareData["yTrain"], squareData["xTest"], squareData["yTest"]
# f = 'square'

learningRate = 0.05
validationFraction = 0
SIGMA = 1
convergenceErrors = []
testErrors = []
nRBF = 20
epochs = 250
nWinners = 9  # 8 eller 9 verkar bra
dataTitles = [
    "RBF CL hard, noisy data",
    "RBF CL soft, noisy data",
]

rbfCLHardNoise = RBFCL(
    learningRate=learningRate,
    validationFraction=validationFraction,
    nRBF=nRBF,
    sigma=SIGMA,
    learningType="sequential",
    positionType="random",
)

rbfCLSoftNoise = RBFCL(
    learningRate=learningRate,
    validationFraction=validationFraction,
    nRBF=nRBF,
    sigma=SIGMA,
    learningType="sequential",
    positionType="random",
    CLType="soft",
    nWinners=nWinners,
)

mseList = []
"""NOISY DATA"""
ballData = dataGenerator.getBallData()
xTrain, yTrain, xTest, yTest = (
    ballData["xTrain"],
    ballData["yTrain"],
    ballData["xTest"],
    ballData["yTest"],
)
"""RBF CL hard noisy data"""
rbfCLHardNoise.fit(
    xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest, epochs=epochs
)
yPred = rbfCLHardNoise.predict(xTest)
err = getMSE(yPred=yPred, yTest=yTest)
testErrors.append(err)
convergenceErrors.append(rbfCLHardNoise.epochErrors)
plotNodes(
   rbf=rbfCLHardNoise, xTrain=xTrain, label=dataTitles[0]
)
mseList.append(rbfCLHardNoise.epochTrainMSEs)
mseList.append(rbfCLHardNoise.epochTestMSEs)

"""RBF cl soft noisy data"""
rbfCLSoftNoise.fit(
    xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest, epochs=epochs
)
yPred = rbfCLSoftNoise.predict(xTest)
err = getMSE(yPred=yPred, yTest=yTest)
print("err: ", err)
testErrors.append(err)
convergenceErrors.append(rbfCLSoftNoise.epochErrors)
plotNodes(
   rbf=rbfCLSoftNoise, xTrain=xTrain, label=dataTitles[1]
)
mseList.append(rbfCLSoftNoise.epochTrainMSEs)
mseList.append(rbfCLSoftNoise.epochTestMSEs)

mseListTitles = [
    "RBF CL hard noisy data train",
    "RBF CL hard noisy data test",
    "RBF CL soft noisy data train",
    "RBF CL soft noisy data test",
]
plotEpochMSEs(mseList, mseListTitles)
print(convergenceErrors)
# convergence plots
plotConvergence(epochs=epochs, errors=convergenceErrors, titles=dataTitles)
# generalisation plots
plotMSE(errors=testErrors, titles=dataTitles)
# node position plots + sinus