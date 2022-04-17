
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
    x = np.arange(4)
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


def plotNodes(rbf, xTest, sinPred, yTest, label):

    plt.figure()
    titleString = "RBF units placements around sinus curve, " + label
    plt.title(titleString)
    centers = rbf.centers
    # print('centers: ', centers)
    y = [0 for i in range(len(centers))]
    plt.scatter(centers, y, c="b", label="RBF centers")
    centersSin = [np.sin(2 * i) for i in centers]
    centersSin = [i[0] for i in centersSin]
    # print('centerssin ', centersSin)
    plt.scatter(centers, centersSin, c="r", label="RBF centers with sin(2x) y value")

    plt.plot(xTest, yTest, label="Real")
    plt.plot(xTest, sinPred, label="Pred")
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

learningRate = 0.1
validationFraction = 0
SIGMA = 1
convergenceErrors = []
testErrors = []
nRBF = 20
epochs = 50
dataTitles = [
    "RBF CL, non-noisy data",
    "Vanilla RBF,non-noisy data",
    "RBF CL, noisy data",
    "Vanilla RBF, noisy data",
]

rbfCLNoNoise = RBFCL(
    learningRate=learningRate,
    validationFraction=validationFraction,
    nRBF=nRBF,
    sigma=SIGMA,
    learningType="sequential",
    positionType="random",
)

rbfCLNoise = RBFCL(
    learningRate=learningRate,
    validationFraction=validationFraction,
    nRBF=nRBF,
    sigma=SIGMA,
    learningType="sequential",
    positionType="random",
)

rbfNoNoise = RBF(
    learningRate=learningRate,
    validationFraction=validationFraction,
    nRBF=nRBF,
    sigma=SIGMA,
    learningType="sequential",
    positionType="random",
)

rbfNoise = RBF(
    learningRate=learningRate,
    validationFraction=validationFraction,
    nRBF=nRBF,
    sigma=SIGMA,
    learningType="sequential",
    positionType="random",
)

mseList = []

"""NON-NOISY DATA"""
sinData = dataGenerator.getSin(noise=False)
xTrain, yTrain, xTest, yTest = (
    sinData["xTrain"],
    sinData["yTrain"],
    sinData["xTest"],
    sinData["yTest"],
)
"""RBF CL non-noisy data"""
rbfCLNoNoise.fit(xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest, epochs=epochs)
yPred = rbfCLNoNoise.predict(xTest)
err = getMSE(yPred=yPred, yTest=yTest)
testErrors.append(err)
convergenceErrors.append(rbfCLNoNoise.epochErrors)
plotNodes(
    rbf=rbfCLNoNoise, xTest=xTest, sinPred=yPred, yTest=yTest, label=dataTitles[0]
)
# print('list:', rbfCLNoNoise.epochTrainMSEs)
mseList.append(rbfCLNoNoise.epochTrainMSEs)
mseList.append(rbfCLNoNoise.epochTestMSEs)

"""RBF non-noisy data"""
rbfNoNoise.fit(xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest, epochs=epochs)
yPred = rbfNoNoise.predict(xTest)
err = getMSE(yPred=yPred, yTest=yTest)
testErrors.append(err)
convergenceErrors.append(rbfNoNoise.epochErrors)
plotNodes(rbf=rbfNoNoise, xTest=xTest, sinPred=yPred, yTest=yTest, label=dataTitles[1])
mseList.append(rbfNoNoise.epochTrainMSEs)
mseList.append(rbfNoNoise.epochTestMSEs)

"""NOISY DATA """
sinData = dataGenerator.getSin(noise=True)
xTrain, yTrain, xTest, yTest = (
    sinData["xTrain"],
    sinData["yTrain"],
    sinData["xTest"],
    sinData["yTest"],
)
"""RBF CL noisy data"""
rbfCLNoise.fit(xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest, epochs=epochs)
yPred = rbfCLNoise.predict(xTest)
err = getMSE(yPred=yPred, yTest=yTest)
testErrors.append(err)
convergenceErrors.append(rbfCLNoise.epochErrors)
plotNodes(rbf=rbfCLNoise, xTest=xTest, sinPred=yPred, yTest=yTest, label=dataTitles[2])
mseList.append(rbfCLNoise.epochTrainMSEs)
mseList.append(rbfCLNoise.epochTestMSEs)

"""RBF noisy data"""
rbfNoise.fit(xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest, epochs=epochs)
yPred = rbfNoise.predict(xTest)
err = getMSE(yPred=yPred, yTest=yTest)
testErrors.append(err)
convergenceErrors.append(rbfNoise.epochErrors)
plotNodes(rbf=rbfNoise, xTest=xTest, sinPred=yPred, yTest=yTest, label=dataTitles[3])
mseList.append(rbfNoise.epochTrainMSEs)
mseList.append(rbfNoise.epochTestMSEs)

mseListTitles = [
    "RBF CL non-noisy data train",
    "RBF CL non-noisy data test",
    "RBF non-noisy data train",
    "RBF non-noisy data test",
    "RBF CL noisy data train",
    "RBF CL noisy data test",
    "RBF noisy data train",
    "RBF noisy data test",
]
plotEpochMSEs(mseList, mseListTitles)
print(convergenceErrors)
# convergence plots
plotConvergence(epochs=epochs, errors=convergenceErrors, titles=dataTitles)
# generalisation plots
plotMSE(errors=testErrors, titles=dataTitles)
# node position plots + sinus

