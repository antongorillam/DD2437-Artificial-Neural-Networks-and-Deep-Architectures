from cProfile import label
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
import dataGenerator 
import math

class SOMTSP:
    
    def __init__(self, eta, nOutput):
        self.eta = eta
        self.W = None
        self.similarityMat = None
        self.nFeatures = 0
        self.nAttributes = 0
        self.nOutput = nOutput

    def fit(self, xInput, nEpochs, maxNeighbourSize=50):
        neighbourSize = maxNeighbourSize 
        # print(f'xInput: {xInput}')
        self.nFeatures, self.nAttributes = xInput.shape
        #print(f'self.nFeatures: {self.nFeatures}')
        indices = np.array([0,1,2,3,4,5,6,7,8,9])
        self.W = np.random.uniform(0, 1, (self.nOutput, self.nAttributes))  
        
        for epoch in range(nEpochs):
            for _, features in enumerate(xInput): 
                # features: row vector with features
                weightIdx = self.calculateSimilarity(features=features)
                print(f'weightIdx: {weightIdx}')
                indxs = np.arange(weightIdx-neighbourSize, weightIdx+neighbourSize)
                weigthsToModify = np.take(indices, indxs, mode='wrap')
                if neighbourSize == 0:
                    weigthsToModify = np.array([weightIdx])
                print(f'weigthsToModify: {weigthsToModify}')
                self.modifyWeights(weightIdxs=weigthsToModify, features=features)
                
                # weightForward = np.array([(weightIdx+i)%self.nFeatures for i in range(neighbourSize+1)] )
                # weightBackward = np.array([(weightIdx-i-1)%self.nFeatures for i in range(neighbourSize)] )
                # self.modifyWeights(weightIdxs=weightBackward, features=features)
                # self.modifyWeights(weightIdxs=weightForward, features=features)

            # print(f'epoch: {epoch}, neighbourSize: {neighbourSize}')
            if epoch < nEpochs/2:
                neighbourSize = 1
            elif epoch == nEpochs-3:
                neighbourSize = 0
            #neighbourSize -= (maxNeighbourSize/nEpochs )
    
    def plotMap(self, xInput, nEpochs, eta):
        pred = []
        for i, features in enumerate(xInput):
            winnerIdx = self.calculateSimilarity(features=features)
            pred.append(winnerIdx)
        
        xCords = xInput[:,0]
        yCords = xInput[:,1]
        plt.scatter(xCords, yCords)
        pred = np.asarray(pred)
        args = np.argsort(pred)
        x=[]
        y=[]

        plt.title(f'Cyclic Tour number of epochs: {nEpochs}, eta: {eta}')
        for i in args:
            x.append(xInput[i][0])
            y.append(xInput[i][1])
        plt.plot(x,y,c='b')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.savefig(f'images/CyclicTour/TSPGraph_epochs{nEpochs}_eta{eta}'.replace('.', ''))
        

    
    def calculateSimilarity(self, features):
        shortestDist = np.inf
        winner = -1
        for weightIdx, weight in enumerate(self.W):
            distance = np.linalg.norm(weight-features)
            #print(f'distance: {distance}, index: {weightIdx}')
            if distance < shortestDist:
                winner = weightIdx
                shortestDist = distance
        return winner

    def modifyWeights(self, weightIdxs, features):
        for weightIdx in weightIdxs:
            self.W[weightIdx] += self.eta * (features - self.W[weightIdx]) 
        

if __name__ == "__main__":
    eta=0.2
    nEpochs=20
    nOutput=10
    inputData = dataGenerator.getCitiesData()
    somTSP = SOMTSP(eta=eta, nOutput=nOutput)
    somTSP.fit(inputData, nEpochs=nEpochs, maxNeighbourSize=2)
    somTSP.plotMap(inputData, nEpochs=nEpochs, eta=eta)
    