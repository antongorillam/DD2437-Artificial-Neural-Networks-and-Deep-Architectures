import math
from re import X

import matplotlib.pyplot as plt
import numpy as np
class TwoLayerPerceptron:

    def __init__(
        self, xInput, targets, numHidden, numOutput, learningRate, alpha, seed=42
    ):
        self.xInput = xInput
        self.targets = targets
        self.numHidden = numHidden
        self.numOutput = numOutput
        self.learningRate = learningRate
        self.alpha = alpha
        self.seed = seed
        # weights init
        numFeatures, numSamples = xInput.shape
        numOutputs, _ = targets.shape 
        # ndim : 2
        # ndata : 200
        # W : numHidden x numFeatures+1 = 3x3    
        self.W = np.random.rand(self.numHidden, numFeatures + 1)
        # V : numOutputs x numHidden+1
        self.V = np.random.rand(numOutputs, self.numHidden + 1)
        self.deltaW = np.zeros(self.W.shape) 
        self.deltaV = np.zeros(self.V.shape)

    def phi(self, x):
        # Phi function [ 2/(1+e**-x) ] - 1
        return (2 / (1 + np.exp(-x))) - 1

    def phiDerivative(self, x):
        # derivative of phi function [ 1 + phi(x) ][ 1 - phi(x) ]/2
        phiRes = self.phi(x)
        # Derivative of phi function [ 1 + phi(x) ][ 1 - phi(x) ]/2
        return ((1 + phiRes) * (1 - phiRes)) / 2

    def forwardPass(self):
        # xInput : numFeatures+1 x ndata
        xInput = np.ones((self.xInput.shape[0] + 1, self.xInput.shape[1]))
        xInput.T[:, :-1] = self.xInput.T[:, :2]
        # hin : numHidden x numFeatures+1 @ numFeatures+1 x ndata = numHidden x ndata 

        res = self.W @ xInput
        hIn = self.phi(self.W @ xInput)
        # hout : numHidden+1 x ndata
        hOut = np.concatenate((hIn, np.ones((1, hIn.shape[1]))), axis=0)
        # O : 1 x numHidden+1 @ numHidden+1 x ndata = 1 x ndata
        oIn = self.phi(self.V @ hOut)
        oOut = self.phi(oIn)
        return hIn, oIn, oOut, hOut

    def backwardPass(self, hIn, hOut, oIn, oOut):

        deltaO = (oOut - self.targets) * (((1 + oOut) * (1 - oOut)) / 2)
        deltaH = (self.V.T @ deltaO) * (((1 + hOut) * (1 - hOut)) / 2) # V.T : numHidden+1 x 1 
        deltaH = deltaH[:self.numHidden]
        return deltaO, deltaH

    def weightUpdate(self, hOut, deltaO, deltaH):

        xInput = np.ones((self.xInput.shape[0] + 1, self.xInput.shape[1]))
        xInput.T[:, :-1] = self.xInput.T[:, :2]

        self.deltaW = (self.alpha * self.deltaW) - ((deltaH @ xInput.T) * (1 - self.alpha)) 
        self.deltaV = (self.alpha * self.deltaV) - ((deltaO @ hOut.T) * (1 - self.alpha))
        
        self.W += self.learningRate * self.deltaW
        self.V += self.learningRate * self.deltaV

    def predict(self, xInput):
        # xInput : numFeatures+1 x ndata
        xInputTemp = np.ones((xInput.shape[0] + 1, xInput.shape[1]))
        xInputTemp.T[:, :-1] = xInput.T[:, :2]
        xInput = xInputTemp
        # hin : numHidden x numFeatures+1 @ numFeatures+1 x ndata = numHidden x ndata 
        res = self.W @ xInput
        hIn = self.phi(self.W @ xInput)
        # hout : numHidden+1 x ndata
        hOut = np.concatenate((hIn, np.ones((1, hIn.shape[1]))), axis=0)
        # O : 1 x numHidden+1 @ numHidden+1 x ndata = 1 x ndata
        oIn = self.phi(self.V @ hOut)
        oOut = self.phi(oIn)
        predictions = list(map(lambda elem: 1 if elem > 0 else -1, oOut[0]))
        return predictions

    def misClassification(self):
        misClass = 0
        _, _, oOut, _ = self.forwardPass()
        for i in range(oOut[0].size):
            y = 1 if oOut[0][i] > 0 else -1
            if y != self.targets[0][i]:
                misClass +=1
        return misClass

    def validationMisclassification(self, validation):

        targets = validation[2:][0]
        validation = validation[:2]
        predictions = self.predict(validation)
        misclassiffied = sum(1 for pred, target in zip(predictions, targets) if pred==target)
        return (len(targets)- misclassiffied) / len(targets)
    
    def mseFunction(self, predictions):
        # 1 X 200 - 1 X 200
        e = self.targets - predictions
        # mse : 1 x n *  n x 1 =  1x1
        mse = (1/e.size) * e @ e.T
        # print('mse test:', mse)
        return np.asscalar(mse)

    def validationMSEFunction(self, validation):

        targets = validation[2:][0]
        validation = validation[:2]
        # xInput : numFeatures+1 x ndata
        xInputTemp = np.ones((validation.shape[0] + 1, validation.shape[1]))
        xInputTemp.T[:, :-1] = validation.T[:, :2]
        xInput = xInputTemp
        # hin : numHidden x numFeatures+1 @ numFeatures+1 x ndata = numHidden x ndata 
        res = self.W @ xInput
        hIn = self.phi(self.W @ xInput)
        # hout : numHidden+1 x ndata
        hOut = np.concatenate((hIn, np.ones((1, hIn.shape[1]))), axis=0)
        # O : 1 x numHidden+1 @ numHidden+1 x ndata = 1 x ndata
        oIn = self.phi(self.V @ hOut)
        oOut = self.phi(oIn)
        e = targets - oOut
        mse = (1/e.size) * e @ e.T

        return np.asscalar(mse)
        
        # batch training
 
    def train(self, epochs, validation=[]):
        numMisclassications = np.array([])
        numValidationMisclassications = np.array([])
        meanSquaredError = [] 
        meanValidationSquaredError = []
        
        for epoch in range(epochs):

            hIn, oIn, oOut, hOut = self.forwardPass()
            deltaO, deltaH = self.backwardPass( hIn=hIn, hOut=hOut, oIn=oIn, oOut=oOut )
            self.weightUpdate( hOut=hOut, deltaO=deltaO, deltaH=deltaH )
            numMisclassications = np.append(numMisclassications, self.misClassification() )
            meanSquaredError.append(self.mseFunction(predictions=oOut))
            # For validations
            if validation != []:
                numValidationMisclassications = np.append( numValidationMisclassications, self.validationMisclassification(validation) )
                meanValidationSquaredError.append(self.validationMSEFunction(validation))
        
        numMisclassications = numMisclassications / self.xInput.shape[1] # Normalize numMisclassications
        return numMisclassications, meanSquaredError, numValidationMisclassications, meanValidationSquaredError
        
    
        

