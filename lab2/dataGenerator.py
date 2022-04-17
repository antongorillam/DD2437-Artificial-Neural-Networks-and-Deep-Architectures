import numpy as np
import matplotlib.pyplot as plt
from math import pi
from random import randint
import pandas as pd
import codecs

def getSin(noise=False):
    """ Generates the sin(2x) data with inteval [0, 2*pi] with 0.
    params
    ------
        noise
            if true, then a noise with 0 mean and 0.1 STD will be added on 

    returns
    -------
        dict with 4 entries:
            xTrain:
                a shape(size, 1) array with training points
            yTrain:
                a shape(size, 1) array with training targets
            xTrain:
                a shape(size, 1) array with test points
            xTrain:
                a shape(size, 1) array with test targets
    """
    xMax = 2*pi
    stepSize = 0.1
    
    xTrain = np.arange(0, xMax, stepSize)
    xTest = np.arange(0+0.05, xMax+0.05, stepSize)
    # Reshaping xTrain and xTest to column vectors
    xTrain = xTrain.reshape(xTrain.size, 1)
    xTest = xTest.reshape(xTest.size, 1)
    # Target
    yTrain = np.sin(2*xTrain)
    yTest = np.sin(2*xTest)
    
    # Add noise to data
    if (noise==True):
        yTrain = np.array([y + np.random.normal(0, 0.1**2) for y in yTrain]).reshape(yTrain.size, 1)
        yTest = np.array([y + np.random.normal(0, 0.1**2) for y in yTest]).reshape(yTest.size, 1)

    sinData = {
        'xTrain' : xTrain,
        'yTrain' : yTrain,
        'xTest' : xTest,
        'yTest' : yTest,
    }
    
    return sinData


def getSquare(noise=False):
    """ Generates the square(2x) data with inteval [0, 2*pi] with 0.
        square(2x) returns 1 if sin(2x) >= 0, else returns -1

        params
    ------
        noise
            if true, then a noise with 0 mean and 0.1 STD will be added on 

    returns
    -------
        dict with 4 entries:
            xTrain:
                a shape(size, 1) array with training points
            yTrain:
                a shape(size, 1) array with training targets
            xTrain:
                a shape(size, 1) array with test points
            xTrain:
                a shape(size, 1) array with test targets
    """
    xMax = 2*pi
    stepSize = 0.1
    
    xTrain = np.arange(0, xMax, stepSize)
    xTest = np.arange(0+0.05, xMax+0.05, stepSize)
    # Reshaping xTrain and xTest to column vectors
    xTrain = xTrain.reshape(xTrain.size, 1)
    xTest = xTest.reshape(xTest.size, 1)
    # Target

    yTrain = np.array([1 if np.sin(2*x)>=0 else -1 for x in xTrain]).reshape(xTrain.size, 1)
    yTest = np.array([1 if np.sin(2*x)>=0 else -1 for x in xTest]).reshape(xTest.size, 1)

    # Add noise to data
    if (noise==True):
        yTrain = np.array([y + np.random.normal(0, 0.1**2) for y in yTrain]).reshape(yTrain.size, 1)
        yTest = np.array([y + np.random.normal(0, 0.1**2) for y in yTest]).reshape(yTest.size, 1)

    squareData = {
        'xTrain' : xTrain,
        'yTrain' : yTrain,
        'xTest' : xTest,
        'yTest' : yTest,
    }
    return squareData

def getBallData():

    # reading csv files for train data
    dataTrain =  pd.read_csv('data/ballist.dat', sep="\t", header=None)
    dfTrain = pd.DataFrame()
    dfTrain[['xAngle','xVel']] = dataTrain[0].str.split(' ', expand=True)
    dfTrain[['yAngle','yVel']] = dataTrain[1].str.split(' ', expand=True)
    dfTrain = dfTrain.apply(pd.to_numeric) 
    xTrain = dfTrain[['xAngle','xVel']].to_numpy()
    yTrain = dfTrain[['yAngle','yVel']].to_numpy() 
    # reading csv files for test data
    dataTest =  pd.read_csv('data/balltest.dat', sep="\t", header=None)
    dfTest = pd.DataFrame()
    dfTest[['xAngle','xVel']] = dataTest[0].str.split(' ', expand=True)
    dfTest[['yAngle','yVel']] = dataTest[1].str.split(' ', expand=True)
    dfTest = dfTest.apply(pd.to_numeric) 
    xTest = dfTest[['xAngle','xVel']].to_numpy()
    yTest = dfTest[['yAngle','yVel']].to_numpy() 
    output = {
        'xTrain' : xTrain,
        'yTrain' : yTrain,
        'xTest' : xTest,
        'yTest' : yTest,
    }
    return output

def getAnimalData():
    file = open('data/animals.dat', 'r')
    for line in file.readlines():
        data = line.rstrip().split(',') #using rstrip to remove the commas
    
    props = np.array([int(i) for i in data]).reshape(32, 84) # We want a 32x84 but data is a 2688 array 
    return props

def getAnimalName():
    ''' returns:
            array with animal names
    '''
    file = open('data/animalnames.txt', 'r')
    animalNames = []
    for i, line in enumerate(file.readlines()):
        line = line.rstrip().split('\t')[0].replace('\'', '')
        animalNames.append(line)
    return np.array(animalNames)

def getCitiesData(dataType='array'):
    ''' returns:
            array with animal names
    '''
    # reading csv files for train data
    df = pd.DataFrame()
    dfTemp =  pd.read_csv('data/cities.dat', sep=";", header=None)
    df[['x','y']] = dfTemp[0].str.split(', ', expand=True)
    df = df.apply(pd.to_numeric) 
    if dataType=='df':
        return df
    elif dataType=='array':
        return df.to_numpy() 
    else: 
        raise Exception(f'"{dataType}" is not a valid dataType, please try "array" or "df"')

def getVotesData():
    ''' returns:
            array with votes for each member, shape: (349, 31) 
            row: member
            col: vote
    '''
    file = open('data/votes.dat', 'r')
    for line in file.readlines():
        data = line.rstrip().split(',') #using rstrip to remove the commas
    votes = np.array([float(vote) for vote in data]).reshape((349, 31))
    return votes

def getParlamentData():
    partyMap = {0:'No Party', 1:'m', 2:'fp', 3:'s', 4:'v', 5:'mp', 6:'kd', 7:'c'}
    df = pd.DataFrame()
    dfNames =  pd.read_csv('data/mpnames.txt', sep=";", header=None, encoding='ISO-8859-1')
    dfNames = dfNames[0].str.split(', ', expand=True)
    dfParties =  pd.read_csv('data/mpparty.dat', header=None, comment='%')
    dfParties = dfParties[0].map(lambda x: partyMap[x])
    dfSex = pd.read_csv('data/mpsex.dat', header=None, comment='%')
    dfSex = dfSex[0].map(lambda x: 'male' if x==0 else 'female')
    dfDistrict = pd.read_csv('data/mpdistrict.dat', header=None, comment='%')
    dfDistrict = dfDistrict[0]
    
    df = pd.DataFrame({
        'x': np.zeros(len(dfNames)),
        'y': np.zeros(len(dfNames)), 
        'name': np.ravel(dfNames.to_numpy()),
        'party': np.ravel(dfParties.to_numpy()),
        'sex': np.ravel(dfSex.to_numpy()),
        'district': np.ravel(dfDistrict.to_numpy()),
    })
    return df

''' Main for testing purposes '''
if __name__ == '__main__':
    df = getParlamentData()
    print(df)