import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from twoLayerPerceptron import TwoLayerPerceptron
import dataGenerator 
import plotFunctions

SEED = 42

def misclassificationPercentage(predA, predB):#prediction, target):
    
    totalCounts = len(predA) + len(predB)
    misclassiffiedA = sum(elem for elem in predA if elem==-1)
    misclassiffiedB = sum(elem for elem in predB if elem==1)
    return (misclassiffiedA + misclassiffiedB) / totalCounts



    # unique, counts = np.unique(prediction, return_counts=True)
    # sampleCounts = dict(zip(unique, counts))
    # if target == -1:
    #     if sampleCounts[target] == len(prediction):
    #         return 1
    #     elif sampleCounts[1] == len(prediction):
    #         return 0
    #     else:
    #         return sampleCounts[target] / (sampleCounts[target] + sampleCounts[1])
        
    # else:
    #     if target == 1:
    #         if sampleCounts[target] == len(prediction):
    #             return 1
    #         elif sampleCounts[-1] == len(prediction):
    #             return 0
    #         else:
    #             return sampleCounts[target] / (sampleCounts[target] + sampleCounts[-1])     

def assignmentPart1():
    NUM_HIDDEN = 10
    NUM_OUTPUT = 1
    LEARNING_RATE = 0.01
    ALPHA = 0.8
    classA, classB = dataGenerator.getData(seed=SEED)
    xInput, target = dataGenerator.addLabels(classA, classB, seed=SEED)

    """
    Modify the number of hidden nodes and demonstrate the effect the size
    of the hidden layer has on the performance (both the mean squared error
    and the number/ratio of misclassiffcations). How many hidden nodes do
    you need to perfectly separate all the available data (if manageable at all
    given your data randomisation)?
    """
    # NODES_TO_TEST = [1, 2, 5, 8, 15, 40]

    # for hiddeNodes in NODES_TO_TEST:
    #     perceptron = TwoLayerPerceptron(
    #         xInput=xInput, 
    #         targets=target, 
    #         numHidden=hiddeNodes, 
    #         numOutput=NUM_OUTPUT, 
    #         learningRate=LEARNING_RATE, 
    #         alpha=ALPHA,
    #         )
    #     misclassifications, meanSquaredError = perceptron.train(epochs=400)

    #     plotFunctions.plotMetric(
    #         metricList=misclassifications,
    #         nHidden=hiddeNodes, 
    #         directory="images/3.1.1_point_1/", 
    #         metric="Misclassification Ratio",
    #         alpha = ALPHA,
    #         learningRate = LEARNING_RATE,
    #         )
    #     plotFunctions.plotMetric(
    #         metricList=meanSquaredError,
    #         nHidden=hiddeNodes, 
    #         directory="images/3.1.1_point_1/", 
    #         metric="Mean Squared Error",
    #         alpha = ALPHA,
    #         learningRate = LEARNING_RATE,
    #         )

    """ 
    Then, formulate a more realistic problem where only a subset of data
    points is available for training a network (data you use to calculate weight
    updates using backprop) and the remaining samples constitute a validation
    dataset for probing generalisation capabilites of the network. To do that,
    subsample the data for training according to the following scenarios: 
    """

    """ Data generation for all samples """

    """ Sample 1: Random 25% from each class """

    """ Sample 2: random 50% from classA """

    """ Sample 3: 20% from a subset of classA for which classA(1,:)<0 and 80% from a subset of classA for which classA(1,:)>0 """

    """ Perceptron training for all samples """

    """ Sample 1: Random 25% from each class """

    """ Sample 2: random 50% from classA """

    """ Sample 3: 20% from a subset of classA for which classA(1,:)<0 and 80% from a subset of classA for which classA(1,:)>0 """

    """ Plot graphs for all samples """

    """ Sample 1: Random 25% from each class """

    """ Sample 2: random 50% from classA """

    """ Sample 3: 20% from a subset of classA for which classA(1,:)<0 and 80% from a subset of classA for which classA(1,:)>0 """


    """ Sample 1: Random 25% from each class """
    classATrain1, classAVal1, classBTrain1, classBVal1 = dataGenerator.dataSample1(classA, classB, seed=SEED)
    # print("sample1: ", classATrain1.shape, classAVal1.shape, classBTrain1.shape, classBVal1.shape)
    xInput1 = np.concatenate((classATrain1.T, classBTrain1.T))
    print("sample1: ", xInput1.shape)
    np.random.shuffle(xInput1)
    target1 = xInput1.T[2:]
    xInput1 = xInput1.T[:2]
    perceptron1 = TwoLayerPerceptron(
            xInput=xInput1, 
            targets=target1, 
            numHidden=NUM_HIDDEN, 
            numOutput=NUM_OUTPUT, 
            learningRate=LEARNING_RATE, 
            alpha=ALPHA,
        )    
    validation1 = np.concatenate((classAVal1.T, classBVal1.T))
    np.random.shuffle(validation1)
    # print(validation1)
    # print
    misclassifications1, meanSquaredError1, validationMisClassification1, validationMeanSquaredError1 = perceptron1.train(epochs=400, validation=validation1.T)
    
    # Plotta misclassifiaction: training vs val
    plotFunctions.plotTrainingAndValidation(
        trainingList=misclassifications1,
        validationList=validationMisClassification1, 
        nHidden=NUM_HIDDEN, 
        directory='images/3.1.1_point_2/sample1/', 
        metricString='Misclassification Rate', 
        alpha=ALPHA, 
        learningRate=LEARNING_RATE,
        )

    plotFunctions.plotTrainingAndValidation(
        trainingList=meanSquaredError1,
        validationList=validationMeanSquaredError1, 
        nHidden=NUM_HIDDEN, 
        directory='images/3.1.1_point_2/sample1/', 
        metricString='Mean Squared Error', 
        alpha=ALPHA, 
        learningRate=LEARNING_RATE,
        )
    
    """ Sample 2: random 50% from classA """
    classATrain2, classAVal2, classB2 = dataGenerator.dataSample2(classA, classB, seed=SEED)
    xInput2 = np.concatenate((classATrain2.T, classB2.T))
    print("sample2: ", xInput2.shape)
    np.random.shuffle(xInput2)

    target2 = xInput2.T[2:]
    xInput2 = xInput2.T[:2]
    perceptron2 = TwoLayerPerceptron(
            xInput=xInput2, 
            targets=target2, 
            numHidden=NUM_HIDDEN, 
            numOutput=NUM_OUTPUT, 
            learningRate=LEARNING_RATE, 
            alpha=ALPHA,
        )

    validation2 = classAVal2.T
    np.random.shuffle(validation2)

    misclassifications2, meanSquaredError2, validationMisClassification2, validationMeanSquaredError2 = perceptron2.train(epochs=400, validation=validation2.T)

    # Plotta misclassifiaction: training vs val
    plotFunctions.plotTrainingAndValidation(
        trainingList=misclassifications2,
        validationList=validationMisClassification2, 
        nHidden=NUM_HIDDEN, 
        directory='images/3.1.1_point_2/sample2/', 
        metricString='Misclassification Rate', 
        alpha=ALPHA, 
        learningRate=LEARNING_RATE,
        )

    plotFunctions.plotTrainingAndValidation(
        trainingList=meanSquaredError2,
        validationList=validationMeanSquaredError2, 
        nHidden=NUM_HIDDEN, 
        directory='images/3.1.1_point_2/sample2/', 
        metricString='Mean Squared Error', 
        alpha=ALPHA, 
        learningRate=LEARNING_RATE,
        )
    # print(f'validati

    # print("sample2: ", classATrain2.shape, classAVal2.shape)
    
    """ Sample 3: 20% from a subset of classA for which classA(1,:)<0 and 80% from a subset of classA for which classA(1,:)>0 """
    classATrainPositive3, classAValPositive3, classATrainNegative3, classAValNegative3 = dataGenerator.dataSample3(classA, seed=SEED)
    # print("sample3: ", classATrainPositive3.shape, classAValPositive3.shape, classATrainNegative3.shape, classAValNegative3.shape)

    # predA3 = perceptron2.predict(classAVal3[:2])
    # totalMisclassifiedSample1 = misclassificationPercentage(predA3, _)



def assignmentPart2():

    return None

def plotData():

    return None


def main():
    assignmentPart1()
    #assignmentPart2()


if __name__ == "__main__":
    main()
