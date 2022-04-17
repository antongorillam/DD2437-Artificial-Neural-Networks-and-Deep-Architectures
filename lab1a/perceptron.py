import matplotlib.pyplot as plt
import numpy as np


class Perceptron:
    def __init__(self, x_input, targets, bias,learningRate=0.01, seed=42, convergeThreshold=0.1, directory=""):
        self.x_input = x_input
        self.targets = targets
        self.bias = bias
        self.weighedSum = 0
        self.learningRate = learningRate 
        self.seed = seed
        np.random.seed(seed)
        self.weights = np.random.rand(1, x_input.shape[0]) 
        self.convergeThreshold = convergeThreshold
        self.directory = directory
    
    def stepFunction(self, val):
        return 1 if val > 0 else 0

    # Online 
    def perceptronLearningOnline(self, maxEpochs):
        
        missClassList = []

        for epoch in range(maxEpochs):
            deltaTot = 0
            missClassEpochCounter = 0
            for x_pair, target in zip(self.x_input.T, self.targets.T):
                
                yPrim = self.weights[0].T @ x_pair
                y = self.stepFunction(yPrim)
                if target - y != 0:
                    missClassEpochCounter += 1
                """
                y - target = 0 if classified correcly
                y - target = 1 if target is 0 but y is classified as 1
                y - target =-1 if target is 1 but y is classified as 0 
                """                
                deltaWeight = self.learningRate * (target-y) * x_pair
                self.weights[0] += deltaWeight
                deltaTot += np.sum(np.abs(deltaWeight))
            missClassList.append(missClassEpochCounter)

            # if deltaTot <= 0:
            #     break
            
        self.plotDecisionBoundary(
            titleString='Perceptron-Learning Online',
            epoch=epoch+1,
            maxEpoch=maxEpochs,
            withBias=self.bias
        )

        return missClassList

    def deltaRuleOnline(self, maxEpochs):
        errors = []
        for epoch in range(maxEpochs):
            deltaTot = 0
            for x_pair, target in zip(self.x_input.T, self.targets.T):
                yPrim = self.weights[0].T @ x_pair
                """
                y - target = 0 if classified correcly
                y - target = 1 if target is 0 but y is classified as 1
                y - target =-1 if target is 1 but y is classified as 0 
                """
                deltaWeight = self.learningRate * (target-yPrim) * x_pair
                self.weights[0] += deltaWeight
                deltaTot += np.sum(np.abs(deltaWeight))

            mse = self.meanSquareError()
            errors.append(mse)
            # print(f"deltaTot: {deltaTot}")
            # if deltaTot < self.convergeThreshold:
            #     break

        self.plotDecisionBoundary(
            titleString='Delta-Rule Online',
            epoch=epoch+1,
            maxEpoch=maxEpochs,
            withBias=self.bias
        )
        return np.array(errors)

    def deltaRuleBatch(self, maxEpochs):
        MSEErrors = []
        for epoch in range(maxEpochs):

            deltaWeight = -self.learningRate * (self.weights @ self.x_input - self.targets) @ self.x_input.T
            self.weights += deltaWeight
            # if np.sum(np.absolute(deltaWeight)) < self.convergeThreshold:
            #     break

            mse = self.meanSquareError()
            MSEErrors.append(mse)
        self.plotDecisionBoundary(
            titleString='Delta Rule Batch',
            epoch=epoch+1,
            maxEpoch=maxEpochs,
            withBias=self.bias
        )
        return MSEErrors

    def meanSquareError(self):
        # e = n x 1 vector ??
        e = self.targets - self.weights[0].T @ self.x_input
        mse = (1/e.size) * e @ e.T
        return mse[0,0]
    
    def predict(self, x_input, learningType):
        predictions = []
        for x_pair in x_input.T:
            x_pair = np.append(x_pair, 1)
            yPrim = self.weights[0].T @ x_pair
            if learningType == "delta_rule":
                y = 1 if yPrim > 0 else -1
            else:
                y = 1 if yPrim > 0 else 0

            predictions.append(y)

        return predictions


    def plotDecisionBoundary(self, titleString, epoch, maxEpoch, withBias=True):
        # Plot datapoints 
        fig = plt.figure()
        x = self.x_input[0]
        y = self.x_input[1]
        
        if withBias:
            dec_bound = (-(self.weights[0,2]/ self.weights[0,1]) / (self.weights[0,2]/ self.weights[0,0])) * x + (-self.weights[0,2] / self.weights[0,1])
        else:
            dec_bound = (-self.weights[0,0] * x) / self.weights[0,1]

        labels = self.targets.reshape(x.size)
        biasString = "with bias" if withBias else "whithout bias"
        
        plt.xlim( min(x)-1, max(x)+1)
        plt.ylim( min(y)-1, max(y)+1)

        xA = [x for x, label in zip(x, labels) if label!=1.0]
        yA = [y for y, label in zip(y, labels) if label!=1.0]
        xB = [x for x, label in zip(x, labels) if label==1.0]
        yB = [y for y, label in zip(y, labels) if label==1.0]
        APlot = plt.scatter(xA, yA, color='g')
        BPlot = plt.scatter(xB, yB, color='b')
        decLine = plt.plot(x, dec_bound)
        plt.grid()
               
        plt.title(f"{titleString}, Epochs: {epoch}/{maxEpoch}, Learning Rate: {self.learningRate}, {biasString}")
        #plt.legend(labels=('class 0','decision boundary' ,'class 1'))
        plt.legend((APlot, BPlot, decLine), ('ClassA', 'ClassB', 'Decision Boundary')
        )
        filename = f"{titleString} {self.learningRate} {biasString}" 
        filename =  ((filename + '.png').replace(" ", "_")).replace(".", "")
        plt.savefig(self.directory+filename)
