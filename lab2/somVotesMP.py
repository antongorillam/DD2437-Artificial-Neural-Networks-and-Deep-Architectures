
from scipy. spatial.distance import cityblock
from matplotlib import image
from sklearn import neighbors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import dataGenerator 
import math

class SOMVotes:
    
    def __init__(self, eta, nOutput):
        self.eta = eta
        self.W = None
        self.similarityMat = None
        self.nFeatures = 0
        self.nAttributes = 0
        self.nOutput = nOutput

    def fit(self, xInput, nEpochs, maxNeighbourSize):
        # np.random.seed(200)
        neighbourSize = maxNeighbourSize 
        self.nFeatures, self.nAttributes = xInput.shape
        self.W = np.random.uniform(0, 1, (self.nOutput, self.nOutput, self.nAttributes)) 

        for epoch in range(nEpochs):
            print(f'Epoch: {epoch}')
            for _, features in enumerate(xInput): 
                # features: row vector with features
                winnerCoord = self.calculateSimilarity(features=features)
                weightToChange = self.getNeighbours(winnerCoord, neighbourSize)
                self.modifyWeights(weightIdxs=weightToChange, features=features)


            neighbourSize -= (maxNeighbourSize/nEpochs )
    

    def getNeighbours(self, winnerCoord, radius):
        neighbours = []
        for i in range(self.nOutput):
            for j in range(self.nOutput):
                possibleNeigbour = np.asarray([i,j])
                dist = np.linalg.norm(possibleNeigbour-winnerCoord)
                if dist <= radius:
                    neighbours.append(possibleNeigbour) 

        return np.asarray(neighbours)
    
    def calculateSimilarity(self, features):
        shortestDist = np.inf
        winner = -1
        for i, wi in enumerate(self.W):
            for j, wj, in enumerate(wi): 
                distance = np.linalg.norm(wj-features)
                # print(f'distance: {distance}')
                if distance < shortestDist:
                    winner = [i, j]
                    shortestDist = distance
        return np.asarray(winner)

    def modifyWeights(self, weightIdxs, features):
        for weightIdx in weightIdxs:
            self.W[weightIdx[0],weightIdx[1]] += self.eta * (features - self.W[weightIdx[0],weightIdx[1]]) 

    def plotMap(self, xInput, df, nEpochs):
        
        for i, features in enumerate(xInput):
            winnerIdx = self.calculateSimilarity(features=features)
            df['x'][i] = winnerIdx[0] + np.random.uniform(-.5,.5) 
            df['y'][i] = winnerIdx[1] + np.random.uniform(-.5,.5) 
        
        # Plot for parties
        titleParty = f'SOM labeled with Parties\nOutputs: {nOutput}, Epochs: {nEpochs}'
        fig, ax = plt.subplots()
        ax = sns.scatterplot(x='x', y='y', hue ='party', data = df, legend='full')
        ax.legend(loc='lower left')
        ax.set_title(titleParty)
        plt.grid()
        titleParty = titleParty.replace(' ', '_')
        titleParty = titleParty.replace(':', '')
        titleParty = titleParty.replace('\n', '_')
        titleParty = titleParty.replace('.', '')
        plt.savefig('images/somVotes/'+titleParty)

        # Plot for sex
        titleGender = f'SOM labeled with Gender\nOutputs: {nOutput}, Epochs: {nEpochs}'
        fig, ax = plt.subplots()
        ax = sns.scatterplot(x='x', y='y', hue ='sex', data = df, legend='full')
        ax.legend(loc='lower left')
        ax.set_title(titleGender)
        plt.grid()
        titleGender = titleGender.replace(' ', '_')
        titleGender = titleGender.replace(':', '')
        titleGender = titleGender.replace('\n', '_')
        titleGender = titleGender.replace('.', '')
        plt.savefig('images/somVotes/'+titleGender)

        # Plot for parties
        titleDistrict = f'SOM labeled with Districts\nOutputs: {nOutput}, Epochs: {nEpochs}'
        fig, ax = plt.subplots()
        ax = sns.scatterplot(x='x', y='y', hue ='district', data = df, legend=False, palette="Paired")
        ax.set_title(titleDistrict)
        plt.grid()
        titleDistrict = titleDistrict.replace(' ', '_')
        titleDistrict = titleDistrict.replace(':', '')
        titleDistrict = titleDistrict.replace('\n', '_')
        titleDistrict = titleDistrict.replace('.', '')
        plt.savefig('images/somVotes/'+titleDistrict)

if __name__ == "__main__":
    eta=0.2
    nEpochs=50   
    nOutput=10
    inputData = dataGenerator.getVotesData()
    parlamentDF = dataGenerator.getParlamentData()
    somTSP = SOMVotes(eta=eta, nOutput=nOutput)
    somTSP.fit(inputData, nEpochs=nEpochs, maxNeighbourSize=5)
    somTSP.plotMap(inputData, parlamentDF, nEpochs)
    