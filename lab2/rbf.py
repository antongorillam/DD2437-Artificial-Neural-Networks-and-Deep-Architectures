import matplotlib.pyplot as plt
import numpy as np
import dataGenerator 
from sklearn.utils import shuffle


class RBF:
    def __init__(
        self,
        learningRate,
        validationFraction,
        nRBF,
        sigma,
        learningType="batch",
        positionType="choice",
    ):
        self.learningRate = learningRate
        self.learningType = learningType
        self.positionType = positionType
        self.validationFraction = validationFraction
        self.nRBF = nRBF
        # self.H = None # Should have dimensions (x_input.size, nRBF) ?
        self.W = np.zeros((nRBF, 1))  # Should have dimensions (x_input.size, nRBF) ?
        self.centers = None
        self.sigma = sigma
        self.epochErrors = None
        self.epochTrainMSEs = None
        self.epochTestMSEs = None

    # 1 g matris
    # 2 center matris
    # 3 andra weights sa blir for hidden
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

        nFeatures = xTrain.shape[1]
        if self.positionType == "choice":
            self.centers = np.zeros((self.nRBF, nFeatures))
            indices = [i for i in range(xTrain.size)]
            np.random.shuffle(indices)
            for i in range(self.nRBF):
                self.centers[i, :] = xTrain[indices[i], :]
            # print('this is the centers:', self.centers)
            # RBF nodes uniformly placed in input space
            # self.centers = np.linspace(0,2*np.pi,num=self.nRBF)

        # place randomly in input space
        if self.positionType == "random":
            self.centers = np.random.rand(self.nRBF, nFeatures)
            for i in range(self.nRBF):
                self.centers[i, :] = self.centers[i, :] * 2 * np.pi

        if self.learningType == "batch":
            self.batchTrain(xTrain, yTrain)
        elif self.learningType == "sequential":
            self.epochErrors = []
            self.epochTrainMSEs = []
            self.epochTestMSEs = []
            for epoch in range(1, epochs):
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
        totalError = sum([np.abs(yP[0] - y[0]) ** 2 for (yP, y) in zip(yPred, yTrain)])
        print(f"totalError: {totalError}")

    def sequentialTrain(self, xTrain, yTrain):
        xTrain, yTrain = shuffle(xTrain, yTrain)
        for xi, x in enumerate(xTrain):
            x = x.reshape((1, 1))
            phiX = self.getG(x)
            phiX = phiX.reshape((phiX.size, 1))

            error = yTrain[xi, 0] - (phiX.T @ self.W)
            error = error.mean()
            deltaW = self.learningRate * error * phiX
            # Update the weights
            self.W += deltaW

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
        G = np.zeros((xs.size, self.nRBF))

        for idxCenter, rbf in enumerate(self.centers):
            for idxData, x in enumerate(xs[:, 0]):
                phi = self.phi(
                    rbf, x
                )  # np.exp((-((x - self.centers[rbf])**2))/(2*self.sigma**2))
                G[idxData, idxCenter] = phi
        return G

    def phi(self, center, dataPoint):
        return np.exp(-self.sigma * np.linalg.norm(center - dataPoint) ** 2)

    def getError(self, yReal, yPred):
        totalError = sum([np.abs(yP[0] - y[0]) ** 2 for (yP, y) in zip(yPred, yReal)])
        return totalError


    def getError(self, yReal, yPred, errorType='MSE'):
        errors = [np.linalg.norm(yP - y) ** 2 for (yP, y) in zip(yPred, yReal)]
        if errorType =='MSE':
            return np.mean(errors)
        else:
            return sum(errors)

""" Main for testing purposes """
if __name__ == "__main__":
    sinData = dataGenerator.getSin()
    xTrain, yTrain, xTest, yTest = (
        sinData["xTrain"],
        sinData["yTrain"],
        sinData["xTest"],
        sinData["yTest"],
    )

    rbf = RBF(
        learningRate=0.1,
        validationFraction=0,
        nRBF=20,
        sigma=1,
        learningType="sequential",
    )
    rbf.fit(xTrain=xTrain, yTrain=yTrain, epochs=200)
    yPred = rbf.predict(xTest)
    plt.figure()
    # plt.ylim(-1.5, 1.5)
    print(yPred)
    plt.plot(xTest, yTest, label='Real')
    plt.plot(xTest, yPred, label='Pred')
    plt.legend()
    plt.grid()
    plt.show()
