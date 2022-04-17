import matplotlib.pyplot as plt
import numpy as np
import dataGenerator 
import math

class SOM:
    
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
        self.W = np.random.uniform(0, 1, (self.nOutput, self.nAttributes)) 

        for epoch in range(nEpochs):
            for _, features in enumerate(xInput): 
                # features: row vector with features
                weightIdx = self.calculateSimilarity(features=features)
                weightToChange = np.array([i for i in range(weightIdx-math.floor(neighbourSize/2), 1+weightIdx+math.floor(neighbourSize/2)) if (i in range(0,self.nOutput))])
                self.modifyWeights(weightIdxs=weightToChange, features=features)

            # print(f'epoch: {epoch}, neighbourSize: {neighbourSize}')
            neighbourSize -= (maxNeighbourSize/nEpochs )
    
    def plotMap(self, xInput, names):
        featMap = {}
        for i, features in enumerate(xInput):
            winnerIdx = self.calculateSimilarity(features=features)
            featMap[names[i]] = winnerIdx
        
        sortedFeat = sorted(featMap.items(), key=lambda x:x[1])
        list(map(lambda x: print(x), sortedFeat)) 
    
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
    
    inputData = dataGenerator.getAnimalData()
    nameData = dataGenerator.getAnimalName()
    som = SOM(eta=0.2, nOutput=100)
    som.fit(inputData, nEpochs = 20)
    som.plotMap(inputData, nameData)