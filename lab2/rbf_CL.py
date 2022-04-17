from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

import dataGenerator
import rbf


class RBFCL:
    def __init__(
        self,
        learningRate,
        validationFraction,
        nRBF,
        sigma,
        learningType="batch",
        positionType="choice",
        CLType="hard",
        nWinners=10,
    ):
        self.learningRate = learningRate
        self.learningType = learningType
        self.positionType = positionType
        self.validationFraction = validationFraction
        self.nRBF = nRBF
        # self.H = None # Should have dimensions (x_input.size, nRBF) ?
        self.W = np.zeros((nRBF, 1))  #
        self.centers = None
        self.sigma = sigma
        self.epochErrors = None
        self.epochTrainMSEs = None
        self.epochTestMSEs = None
        self.CLType = CLType
        self.nWinners = nWinners

    # 1 g matris
    # 2 center matris
    # 3 andra weights sa blir for hidden

    def placeCenters(self, xTrain):
        nFeatures = xTrain.shape[1]
        self.W = np.zeros((self.nRBF, nFeatures))
        # place randomly in input space
        self.centers = np.random.rand(self.nRBF, nFeatures)
        for i in range(self.nRBF):
            self.centers[i, :] = self.centers[i, :] * 2 * np.pi

    def fit(self, xTrain, yTrain, xTest, yTest, epochs=50):
        """ Trains the model with labeled training data 

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : returns a trained MLP model.

        """
        # place centers
        self.placeCenters(xTrain=xTrain)

        if self.learningType == "batch":
            self.batchTrain(xTrain, yTrain)
        elif self.learningType == "sequential":
            self.epochErrors = []
            self.epochTrainMSEs = []
            self.epochTestMSEs = []
            for epoch in range(1, epochs):
                xTrain, yTrain = shuffle(xTrain, yTrain)
                if self.CLType == "hard":
                    self.competitiveLearningHard(xTrain=xTrain)
                if self.CLType == "soft":
                    self.competitiveLearningSoft(xTrain=xTrain, n=self.nWinners)
                self.sequentialTrain(xTrain, yTrain)
                yPred = self.predict(xTrain)
                yPredTest = self.predict(xTest)
                self.epochErrors.append(self.getError(yTrain, yPred))
                self.epochTestMSEs.append(self.getError(yTrain, yPred))
                self.epochTrainMSEs.append(self.getError(yTest, yPredTest))
        else:
            raise Exception(
                f'"{self.learningType}" is not a valid learningType, please try "batch" or "sequential"'
            )

    def batchTrain(self, xTrain, yTrain):
        G = self.getG(xTrain)
        # self.W = np.linalg.pinv(G) @ yTrain
        self.W = np.dot(np.linalg.pinv(G), yTrain)

        yPred = self.predict(xTrain)
        totalError = sum([np.abs(yP - y) ** 2 for (yP, y) in zip(yPred, yTrain)])
        print(f"totalError: {totalError}")

    def sequentialTrain(self, xTrain, yTrain):
        # xTrain, yTrain = shuffle(xTrain, yTrain)
        nSample = xTrain.shape[0]
        nFeature = xTrain.shape[1]
        for xi, x in enumerate(xTrain):
            x = x.reshape((1, nFeature))
            phiX = self.getG(x)
            # print('phiX.shape: ', phiX.shape)
            # print('self.W.shape: ', self.W.shape)
            # print('yTrain[xi, :].shape: ', yTrain[xi, :].shape)
            phiX = phiX.reshape((phiX.size, 1))
            error = yTrain[xi, :] - (phiX.T @ self.W)
            error = error.mean()
            deltaW = self.learningRate * error * phiX
            # Update the weights
            self.W += deltaW

    def competitiveLearningHard(self, xTrain):

        for xi, x in enumerate(xTrain):
            # x = x.reshape((1, 1))
            # phiX = self.getG(x)
            # for all check closest center
            winner = -1
            distance = np.infty
            for i, node in enumerate(self.centers):
                # newDistance = abs(np.norm(np.asarray(x, np.sin(2*x)) - np.asarray(node, np.sin(2*node))))
                newDistance = np.linalg.norm(x - node)
                if newDistance < distance:
                    winner = i
                    distance = newDistance
            # update center
            self.centers[winner, :] += self.learningRate * (x - self.centers[winner, :])

    def competitiveLearningSoft(self, xTrain, n):

        for _, xCord in enumerate(xTrain):
            # x = x.reshape((1, 1))
            # phiX = self.getG(x)
            # for all check closest center
            distanceDict = {}
            for centerIdx, centerCord in enumerate(self.centers):
                # newDistance = abs(np.norm(np.asarray(x, np.sin(2*x)) - np.asarray(unit, np.sin(2*unit))))
                newDistance = np.linalg.norm(xCord - centerCord)
                distanceDict[centerIdx] = newDistance
            k = Counter(distanceDict)

            bottomN = k.most_common()[: -n - 1 : -1]
            # closest, 2d closest, 3d closest ...
            # print("distanceDict: ", distanceDict)
            # print("bottomN: ", bottomN)

            # update n-closest centers
            for i, node in enumerate(bottomN):
                # (i/n)**-1 is the normalizing factor
                normalizingFactor = 1 - i / n
                self.centers[node[0], :] += (
                    normalizingFactor
                    * self.learningRate
                    * (xCord - self.centers[node[0], :])
                )

    def predict(self, x):

        G = self.getG(x)
        # yPred = G @ self.W
        yPred = np.dot(G, self.W)
        return yPred

    def getG(self, xs):
        """ 
        params:
            x : ndarray of shape (n_samples, n_features)
            The input data.
        return:
            G : ndarray of shape (n_samples, nRBFs)
            The interpolation matrix
        """
        nSample = xs.shape[0]
        nFeature = xs.shape[1]
        G = np.zeros((nSample, self.nRBF))

        for idxCenter, rbf in enumerate(self.centers):
            for idxData, x in enumerate(xs[:,]):
                phi = self.phi(
                    rbf, x
                )  # np.exp((-((x - self.centers[rbf])**2))/(2*self.sigma**2))
                G[idxData, idxCenter] = phi
        return G

    def phi(self, center, dataPoint):
        return np.exp(-self.sigma * np.linalg.norm(center - dataPoint) ** 2)

    def getError(self, yReal, yPred, errorType='MSE'):
        '''
        returns a scalar error, dependent on yReal and yPred
        '''
        errors = [np.linalg.norm(yP - y) ** 2 for (yP, y) in zip(yPred, yReal)]
        if errorType == "MSE":
            return np.mean(errors)
        elif errorType=='TotalError':
            return sum(errors)


""" Main for testing purposes """
if __name__ == "__main__":
    ballData = dataGenerator.getBallData()
    xTrain, yTrain, xTest, yTest = (
        ballData["xTrain"],
        ballData["yTrain"],
        ballData["xTest"],
        ballData["yTest"],
    )

    # sinData = dataGenerator.getSin(noise=True)
    # xTrain, yTrain, xTest, yTest = (
    # sinData["xTrain"],
    # sinData["yTrain"],
    # sinData["xTest"],
    # sinData["yTest"],
    # )

    rbf = RBFCL(
        learningRate=0.1,
        validationFraction=0,
        nRBF=20,
        sigma=1,
        learningType="sequential",
        CLType="hard",
    )
    rbf.fit(xTrain=xTrain, yTrain=yTrain, xTest=xTest, yTest=yTest, epochs=200)
    yPred = rbf.predict(xTest)
    error = rbf.getError(yTrain, yPred, errorType="MSE")
    print(f"Error is {error}")
    # plt.figure()
    # plt.ylim(-1.5, 1.5)
    # plt.plot(xTest, yTest, label='Real')
    # plt.plot(xTest, yPred, label='Pred')
    # plt.legend()
    # plt.grid()
    # plt.show()
