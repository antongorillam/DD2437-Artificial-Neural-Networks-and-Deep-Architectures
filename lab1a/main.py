import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron


def plotData(classA, classB):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(classA[0, :], classA[1, :], s=10, c="b", marker="s", label="classA")
    ax1.scatter(classB[0, :], classB[1, :], s=10, c="r", marker="o", label="classB")
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


def testPlotData(x_input, labels):
    fig = plt.figure()
    x = x_input[0]
    y = x_input[1]
    labels = labels.reshape(x.size)
    plt.scatter(x, y, c=labels, cmap="winter")
    plt.grid()
    plt.show()


def generateData(N, mA, mB, sigmaA, sigmaB, seed=100):
    classA = np.zeros((2, N))  # init
    classB = np.zeros((2, N))  # init
    np.random.seed(seed)

    classA[0] = np.random.normal(mA[0], sigmaA, N)
    classA[1] = np.random.normal(mA[1], sigmaA, N)
    classB[0] = np.random.normal(mB[0], sigmaB, N)
    classB[1] = np.random.normal(mB[1], sigmaB, N)

    return classA, classB


def generateData2():
    N = 100
    mA = [1, 0, 0.3]
    mB = [0.0, -0.1]
    sigmaA = 0.2
    sigmaB = 0.3

    classA = np.zeros((2, N))  # init
    classB = np.zeros((2, N))  # init
    np.random.seed(42)

    classA[0] = np.concatenate(
        (
            np.random.normal(-mA[0], sigmaA, round(0.5 * N)),
            np.random.normal(+mA[0], sigmaA, round(0.5 * N)),
        )
    )
    classA[1] = np.random.normal(mA[1], sigmaA, N)
    classB[0] = np.random.normal(mB[0], sigmaB, N)
    classB[1] = np.random.normal(mB[1], sigmaB, N)

    return classA, classB


def addLabels(classA, classB, learning="perceptron_learning", seed=100, bias=True):
    """ Add labels and concatinates the classas
    classA: extra column with 1:s will be added
    classB: extra column with 0:s will be added
    
    return: labeled
    """
    np.random.seed(seed)

    numRowsA = classA.shape[1]
    zeroCol = np.zeros((1, numRowsA))

    numRowsB = classB.shape[1]
    onesCol = np.ones((1, numRowsB))
    negOnesCol = -1 * np.ones((1, numRowsA))

    # if perceptron
    if learning == "perceptron_learning":
        labeledA = np.append(classA, zeroCol, axis=0)
        labeledB = np.append(classB, onesCol, axis=0)

    # else
    elif learning == "delta_rule":
        labeledA = np.append(classA, negOnesCol, axis=0)
        labeledB = np.append(classB, onesCol, axis=0)
    else:
        raise Exception(f"{learning} is not a valid learning method")

    labeledAB = np.append(labeledA, labeledB, axis=1).T
    np.random.shuffle(labeledAB)  # Shuffles the classes
    target = labeledAB[:, 2:3]
    x_input = np.ones((labeledAB.shape[0], labeledAB.shape[1]))
    x_input[:, :-1] = labeledAB[:, :2]
    return x_input.T, target.T


def compareOnline(classA, classB, directory):
    """
    Apply and compare perceptron learning with the delta learning rule in
    online (sequential) mode on the generated dataset. Adjust the learning
    rate and study the convergence of the two algorithms.
    """
    CONVERGETHESHOLD = 0.01
    EPOCHS = 50
    LEARNING_RATES = [0.0001, 0.001, 0.005, 0.01, 0.1]
    x_inputPerceptronLearning, targetsPerceptronLearning = addLabels(
        classA, classB, learning="perceptron_learning"
    )
    x_inputDeltaRule, targetsDeltaRule = addLabels(
        classA, classB, learning="delta_rule"
    )

    for learningRate in LEARNING_RATES:
        """ Create object for Perceptron Learning """
        perceptronPerceptronLearningOnline = Perceptron(
            x_input=x_inputPerceptronLearning,
            targets=targetsPerceptronLearning,
            bias=True,
            learningRate=learningRate,
            convergeThreshold=CONVERGETHESHOLD,
            seed=170,
            directory=directory,
        )
        perceptronPerceptronLearningOnline.perceptronLearningOnline(maxEpochs=EPOCHS)
        """ Create object for Delta Learning """
        perceptronDeltaRuleOnline = Perceptron(
            x_input=x_inputDeltaRule,
            targets=targetsDeltaRule,
            bias=True,
            learningRate=learningRate,
            convergeThreshold=CONVERGETHESHOLD,
            seed=170,
            directory=directory,
        )
        perceptronDeltaRuleOnline.deltaRuleOnline(maxEpochs=EPOCHS)


def compareOnlineToBatch(classA, classB):
    """
    Compare sequential with a batch learning approach for the delta rule. How
    quickly (in terms of epochs) do the algorithms converge? Please adjust the
    learning rate and plot the learning curves for each variant. Bear in mind
    that for sequential learning you should not use the matrix form of the lear-
    ning rule discussed in section 2.2 and instead perform updates iteratively
    for each sample. How sensitive is learning to random initialisation?
    """
    EPOCHS = 50
    LEARNING_RATES = [0.0001, 0.001, 0.005, 0.01, 0.1]
    x_inputDeltaRule, targetsDeltaRule = addLabels(
        classA, classB, learning="delta_rule"
    )

    for learningRate in LEARNING_RATES:
        """ Create object for Delta-Rule batch learning """
        perceptronDeltaRuleBatch = Perceptron(
            x_input=x_inputDeltaRule,
            targets=targetsDeltaRule,
            bias=True,
            learningRate=learningRate,
            directory="images/compareOnlineToBatch/",
        )
        batchErrors = perceptronDeltaRuleBatch.deltaRuleBatch(maxEpochs=EPOCHS)

        """ Create object for Delta Learning online learning """
        perceptronDeltaRuleOnline = Perceptron(
            x_input=x_inputDeltaRule,
            targets=targetsDeltaRule,
            bias=True,
            learningRate=learningRate,
            directory="images/compareOnlineToBatch/",
        )
        onlineErrors = perceptronDeltaRuleOnline.deltaRuleOnline(maxEpochs=EPOCHS)

        plotMSE(
            errorsA=batchErrors,
            learningTypeA="Batch Learning",
            errorsB=onlineErrors,
            learningTypeB="Online Learning",
            learningRate=learningRate,
            withBias=True,
            directory="images/compareOnlineToBatch/",
        )


def compareRemoveBias(classA, classB, classC, classD):
    """
    Remove the bias, train your network with the delta rule in batch mo-
    de and test its behaviour. In what cases would the perceptron without
    bias converge and classify correctly all data samples? Please verify your
    hypothesis by adjusting data parameters, mA and mB.
    """
    EPOCHS = 50
    LEARNING_RATES = [0.0001, 0.001, 0.005, 0.01, 0.1]
    x_inputDeltaRule, targetsDeltaRule = addLabels(
        classA, classB, learning="delta_rule"
    )
    x_inputDeltaRule = x_inputDeltaRule[
        :-1, :
    ]  # Removes the bias, which is on the last row

    """ Create object for Delta-Rule batch learning """
    for learningRate in LEARNING_RATES:
        perceptronDeltaRuleBatch = Perceptron(
            x_input=x_inputDeltaRule,
            targets=targetsDeltaRule,
            bias=False,
            learningRate=learningRate,
            directory="images/compareRemoveBias/",
        )
        perceptronDeltaRuleBatch.deltaRuleBatch(maxEpochs=EPOCHS)

    x_inputDeltaRule, targetsDeltaRule = addLabels(
        classC, classD, learning="delta_rule"
    )
    x_inputDeltaRule = x_inputDeltaRule[
        :-1, :
    ]  # Removes the bias, which is on the last row
    for learningRate in LEARNING_RATES:
        perceptronDeltaRuleBatch = Perceptron(
            x_input=x_inputDeltaRule,
            targets=targetsDeltaRule,
            bias=False,
            learningRate=learningRate,
            directory="images/compareRemoveBiasOrigo/",
        )
        perceptronDeltaRuleBatch.deltaRuleBatch(maxEpochs=EPOCHS)


def plotMSE(
    errorsA, learningTypeA, errorsB, learningTypeB, learningRate, withBias, directory
):
    biasString = "with bias" if withBias else "without bias"
    titleString = f"MSE for {learningTypeA} vs {learningTypeB}\nWith Learning Rate {learningRate} and {biasString}"
    plt.figure(figsize=(15, 10), dpi=100)
    plt.plot(errorsA, label=learningTypeA)
    plt.plot(errorsB, label=learningTypeB)
    plt.title(titleString)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.legend()
    plt.grid()
    filename = titleString
    filename.replace(" ", "_")
    filename = filename + ".png"
    plt.savefig(directory + filename)


def assignmentPart1():

    Ndata = 100
    mALinSep = [1.5, 2.0]
    mBLinSep = [-1.0, 0.0]
    sigmaALinSep = 0.5
    sigmaBLinSep = 0.5
    classA, classB = generateData(
        N=Ndata, mA=mALinSep, mB=mBLinSep, sigmaA=sigmaALinSep, sigmaB=sigmaBLinSep,
    )

    """
    T ~= weights * x_input
    x_input.shape (3, 200)
    target.shape  (1, 200)
    weights.shape (1, 3)
    """
    compareOnline(classA, classB, "images/compareOnline/")
    compareOnlineToBatch(classA, classB)

    mCLinSep = [1.2, 1.2]
    mDLinSep = [-1.2, -1.2]
    sigmaCLinSep = 0.5
    sigmaDLinSep = 0.5
    classC, classD = generateData(
        N=Ndata, mA=mCLinSep, mB=mDLinSep, sigmaA=sigmaCLinSep, sigmaB=sigmaDLinSep,
    )
    compareRemoveBias(classA, classB, classC, classD)



def randomSubarray(array, remainPart, seed=100):
    np.random.seed(seed)

    noSamples = array.shape[1]
    keepSize = int(noSamples * remainPart)
    keepIndices = np.sort(
        np.random.choice(range(noSamples), size=keepSize, replace=False)
    )

    subArr = np.zeros((2, keepSize))

    for i in range(keepSize):
        subArr[0, i] = array[0, keepIndices[i]]
        subArr[1, i] = array[1, keepIndices[i]]

    return subArr


def getSample4(classA, classB):
    # first get index of those fitting

    noElemsToDelete = int(0.5*classA.shape[0])
    noCond1ElemsToDelete = int(0.2*noElemsToDelete)
    noCond2ElemsToDelete = int(0.8*noElemsToDelete)

    cond1Indices = [int(i) for i, _ in enumerate(classA[1, :]) if classA[1, i] < 0]

    cond2Indices = [int(i) for i, _ in enumerate(classA[1, :]) if classA[1, i] > 0]
    

    #cond1DeleteSize = int(0.2 * len(cond1Indices))
    cond1IndicicesToDelete = np.sort(
        np.random.choice(cond1Indices, size=noCond1ElemsToDelete, replace=False)
    )

    #cond2DeleteSize = int(0.8 * len(cond2Indices))
    cond2IndicicesToDelete = np.sort(
        np.random.choice(cond2Indices, size=noCond2ElemsToDelete, replace=False)
    )

    indicesToDelete = np.sort(
        np.concatenate([cond1IndicicesToDelete, cond2IndicicesToDelete])
    )
    #print(indicesToDelete.shape)
    indicesToKeep = int(classA[0, :].size - len(indicesToDelete))

    classASample4 = np.zeros((2, indicesToKeep))

    i = 0
    for j in range(classA[1, :].size):
        if j not in indicesToDelete:
            classASample4[0, i] = classA[0, j]
            classASample4[1, i] = classA[1, j]
            i += 1

    classBSample4 = classB

    return classASample4, classBSample4


def getSensitivityAndspecificity(classA, classB, predicitionA, predicitionB, learningType):

    unique, counts = np.unique(predicitionA, return_counts=True)
    sample1Counts = dict(zip(unique, counts))

    if learningType == "delta_rule":
        sample1ClassACorrectCount = sample1Counts[-1] if -1 in sample1Counts else 0 
        unique, counts = np.unique(predicitionB, return_counts=True)
        sample1Counts = dict(zip(unique, counts))
        sample1ClassBCorrectCount = sample1Counts[1] if 1 in sample1Counts else 0
    else:
        sample1ClassACorrectCount = sample1Counts[0] if 0 in sample1Counts else 0 
        unique, counts = np.unique(predicitionB, return_counts=True)
        sample1Counts = dict(zip(unique, counts))
        sample1ClassBCorrectCount = sample1Counts[1] if 1 in sample1Counts else 0

    sample1ClassACount = classA.shape[1]
    sample1ClassBCount = classB.shape[1]
    """ specificity and sensitivity """
    # classA = neg, classB = pos
    # sensitivity = tpr = TN / TN 
    sensitivity = sample1ClassBCorrectCount / sample1ClassBCount
    # specificity = tnr
    specificity = sample1ClassACorrectCount / sample1ClassACount
    return sensitivity, specificity

def deltaSubSampling():
    
    """ 3.1.3 Classification of samples that are not linearly separable """

    """ 3.1.3. Generating the datasets """
    classA, classB = generateData2()

    # sampling 1:
    classASample1 = randomSubarray(array=classA, remainPart=0.75)
    classBSample1 = randomSubarray(array=classB, remainPart=0.75)

    # sampling 2: random 50% from classA
    classASample2 = randomSubarray(array=classA, remainPart=0.50)
    classBSample2 = classB
    # sampling 3: random 50% from classB
    classASample3 = classA
    classBSample3 = randomSubarray(array=classB, remainPart=0.50)

    # sampling 4
    classASample4, classBSample4 = getSample4(classA, classB)

    """ train on new dataset and get regular performance measuerments """
    x_inputSample0, targetsSample0 = addLabels(classA, classB, learning="delta_rule")

    x_inputSample1, targetsSample1 = addLabels(
        classASample1, classBSample1, learning="delta_rule"
    )

    x_inputSample2, targetsSample2 = addLabels(
        classASample2, classBSample2, learning="delta_rule"
    )

    x_inputSample3, targetsSample3 = addLabels(
        classASample3, classBSample3, learning="delta_rule"
    )

    x_inputSample4, targetsSample4 = addLabels(
        classASample4, classBSample4, learning="delta_rule"
    )

    LEARNING_RATE = 0.001
    EPOCHS = 50

    peceptronSample0 = Perceptron(
        x_input=x_inputSample0,
        targets=targetsSample0,
        bias=True,
        learningRate=LEARNING_RATE,
        directory="images/sample0/",
    )

    peceptronSample1 = Perceptron(
        x_input=x_inputSample1,
        targets=targetsSample1,
        bias=True,
        learningRate=LEARNING_RATE,
        directory="images/sample1/",
    )

    peceptronSample2 = Perceptron(
        x_input=x_inputSample2,
        targets=targetsSample2,
        bias=True,
        learningRate=LEARNING_RATE,
        directory="images/sample2/",
    )

    peceptronSample3 = Perceptron(
        x_input=x_inputSample3,
        targets=targetsSample3,
        bias=True,
        learningRate=LEARNING_RATE,
        directory="images/sample3/",
    )

    peceptronSample4 = Perceptron(
        x_input=x_inputSample4,
        targets=targetsSample4,
        bias=True,
        learningRate=LEARNING_RATE,
        directory="images/sample4/",
    )

    """ get sensitivity and specificity for all samples """
    """ sample 0 """

    peceptronSample0.deltaRuleBatch(maxEpochs=EPOCHS)

    predictionASample0 = peceptronSample0.predict(classA, learningType = "delta_rule")
    predictionBSample0 = peceptronSample0.predict(classB, learningType = "delta_rule")
    sensitivitySample0, specificitySample0 = getSensitivityAndspecificity(
        classA=classASample1,
        predicitionA=predictionASample0,
        classB=classBSample1,
        predicitionB=predictionBSample0,
        learningType="delta_rule",
    )
    print(
        f"sensitivitySample0: {sensitivitySample0}, specificitySample0: {specificitySample0}"
    )

    """ sample 1 """
    peceptronSample1.deltaRuleBatch(maxEpochs=EPOCHS)
    predictionASample1 = peceptronSample1.predict(classASample1, learningType = "delta_rule")
    predictionBSample1 = peceptronSample1.predict(classBSample1, learningType = "delta_rule")
    sensitivitySample1, specificitySample1 = getSensitivityAndspecificity(
        classA=classASample1,
        predicitionA=predictionASample1,
        classB=classBSample1,
        predicitionB=predictionBSample1,
        learningType="delta_rule",
    )

    """ sample 2 """
    peceptronSample2.deltaRuleBatch(maxEpochs=EPOCHS)
    predictionASample2 = peceptronSample2.predict(classASample2, learningType = "delta_rule")
    predictionBSample2 = peceptronSample2.predict(classBSample2, learningType = "delta_rule")
    sensitivitySample2, specificitySample2 = getSensitivityAndspecificity(
        classA=classASample2,
        predicitionA=predictionASample2,
        classB=classBSample2,
        predicitionB=predictionBSample2,
        learningType="delta_rule",
    )

    """ sample 3 """
    peceptronSample3.deltaRuleBatch(maxEpochs=EPOCHS)
    predictionASample3 = peceptronSample3.predict(classASample3, learningType = "delta_rule")
    predictionBSample3 = peceptronSample3.predict(classBSample3, learningType = "delta_rule")
    sensitivitySample3, specificitySample3 = getSensitivityAndspecificity(
        classA=classASample3,
        predicitionA=predictionASample3,
        classB=classBSample3,
        predicitionB=predictionBSample3,
        learningType="delta_rule",
    )

    """ sample 4 """
    peceptronSample4.deltaRuleBatch(maxEpochs=EPOCHS)
    predictionASample4 = peceptronSample4.predict(classASample4, learningType = "delta_rule")
    predictionBSample4 = peceptronSample4.predict(classBSample4, learningType = "delta_rule")
    sensitivitySample4, specificitySample4 = getSensitivityAndspecificity(
        classA=classASample4,
        predicitionA=predictionASample4,
        classB=classBSample4,
        predicitionB=predictionBSample4,
        learningType="delta_rule",
    )
    print(f"Delta-Rule, with Learning Rate of {LEARNING_RATE} and {EPOCHS} epochs")
    print(
        f"sensitivitySample0: {sensitivitySample0}, specificitySample0: {specificitySample0}"
    )
    print(
        f"sensitivitySample1: {sensitivitySample1}, specificitySample1: {specificitySample1}"
    )
    print(
        f"sensitivitySample2: {sensitivitySample2}, specificitySample2: {specificitySample2}"
    )
    print(
        f"sensitivitySample3: {sensitivitySample3}, specificitySample3: {specificitySample3}"
    )
    print(
        f"sensitivitySample4: {sensitivitySample4}, specificitySample4: {specificitySample4}"
    )

def perceptronSubSampling():

    """ 3.1.3 Classification of samples that are not linearly separable """

    """ 3.1.3. Generating the datasets """
    classA, classB = generateData2()

    # sampling 1:
    classASample1 = randomSubarray(array=classA, remainPart=0.75)
    classBSample1 = randomSubarray(array=classB, remainPart=0.75)

    # sampling 2: random 50% from classA
    classASample2 = randomSubarray(array=classA, remainPart=0.50)
    classBSample2 = classB
    # sampling 3: random 50% from classB
    classASample3 = classA
    classBSample3 = randomSubarray(array=classB, remainPart=0.50)

    # sampling 4
    classASample4, classBSample4 = getSample4(classA, classB)

    """ train on new dataset and get regular performance measuerments """
    x_inputSample0, targetsSample0 = addLabels(
        classA, classB, learning="perceptron_learning"
    )

    x_inputSample1, targetsSample1 = addLabels(
        classASample1, classBSample1, learning="perceptron_learning"
    )

    x_inputSample2, targetsSample2 = addLabels(
        classASample2, classBSample2, learning="perceptron_learning"
    )

    x_inputSample3, targetsSample3 = addLabels(
        classASample3, classBSample3, learning="perceptron_learning"
    )

    x_inputSample4, targetsSample4 = addLabels(
        classASample4, classBSample4, learning="perceptron_learning"
    )

    LEARNING_RATE = 0.001
    EPOCHS = 50

    peceptronSample0 = Perceptron(
        x_input=x_inputSample0,
        targets=targetsSample0,
        bias=True,
        learningRate=LEARNING_RATE,
        directory="images/sample0/",
    )

    peceptronSample1 = Perceptron(
        x_input=x_inputSample1,
        targets=targetsSample1,
        bias=True,
        learningRate=LEARNING_RATE,
        directory="images/sample1/",
    )

    peceptronSample2 = Perceptron(
        x_input=x_inputSample2,
        targets=targetsSample2,
        bias=True,
        learningRate=LEARNING_RATE,
        directory="images/sample2/",
    )

    peceptronSample3 = Perceptron(
        x_input=x_inputSample3,
        targets=targetsSample3,
        bias=True,
        learningRate=LEARNING_RATE,
        directory="images/sample3/",
    )

    peceptronSample4 = Perceptron(
        x_input=x_inputSample4,
        targets=targetsSample4,
        bias=True,
        learningRate=LEARNING_RATE,
        directory="images/sample4/",
    )

    """ get sensitivity and specificity for all samples """
    """ sample 0 """

    peceptronSample0.perceptronLearningOnline(maxEpochs=EPOCHS)

    predictionASample0 = peceptronSample0.predict(classA, learningType="perceptron_learning")
    predictionBSample0 = peceptronSample0.predict(classB, learningType="perceptron_learning")
    sensitivitySample0, specificitySample0 = getSensitivityAndspecificity(
        classA=classASample1,
        predicitionA=predictionASample0,
        classB=classBSample1,
        predicitionB=predictionBSample0,
        learningType="perceptron_learning",
    )

    """ sample 1 """
    peceptronSample1.perceptronLearningOnline(maxEpochs=EPOCHS)
    predictionASample1 = peceptronSample1.predict(classASample1, learningType="perceptron_learning")
    predictionBSample1 = peceptronSample1.predict(classBSample1, learningType="perceptron_learning")
    sensitivitySample1, specificitySample1 = getSensitivityAndspecificity(
        classA=classASample1,
        predicitionA=predictionASample1,
        classB=classBSample1,
        predicitionB=predictionBSample1,
        learningType="perceptron_learning",
    )

    """ sample 2 """
    peceptronSample2.perceptronLearningOnline(maxEpochs=EPOCHS)
    predictionASample2 = peceptronSample2.predict(classASample2, learningType="perceptron_learning")
    predictionBSample2 = peceptronSample2.predict(classBSample2, learningType="perceptron_learning")
    sensitivitySample2, specificitySample2 = getSensitivityAndspecificity(
        classA=classASample2,
        predicitionA=predictionASample2,
        classB=classBSample2,
        predicitionB=predictionBSample2,
        learningType="perceptron_learning",
    )

    """ sample 3 """
    peceptronSample3.perceptronLearningOnline(maxEpochs=EPOCHS)
    predictionASample3 = peceptronSample3.predict(classASample3, learningType="perceptron_learning")
    predictionBSample3 = peceptronSample3.predict(classBSample3, learningType="perceptron_learning")
    sensitivitySample3, specificitySample3 = getSensitivityAndspecificity(
        classA=classASample3,
        predicitionA=predictionASample3,
        classB=classBSample3,
        predicitionB=predictionBSample3,
        learningType="perceptron_learning",
    )

    """ sample 4 """
    peceptronSample4.perceptronLearningOnline(maxEpochs=EPOCHS)
    predictionASample4 = peceptronSample4.predict(classASample4, learningType="perceptron_learning")
    predictionBSample4 = peceptronSample4.predict(classBSample4, learningType="perceptron_learning")
    sensitivitySample4, specificitySample4 = getSensitivityAndspecificity(
        classA=classASample4,
        predicitionA=predictionASample4,
        classB=classBSample4,
        predicitionB=predictionBSample4,
        learningType="perceptron_learning",
    )

    print(f"Delta-Rule, with Learning Rate of {LEARNING_RATE} and {EPOCHS} epochs")
    print(
        f"sensitivitySample0: {sensitivitySample0}, specificitySample0: {specificitySample0}"
    )
    print(
        f"sensitivitySample1: {sensitivitySample1}, specificitySample1: {specificitySample1}"
    )
    print(
        f"sensitivitySample2: {sensitivitySample2}, specificitySample2: {specificitySample2}"
    )
    print(
        f"sensitivitySample3: {sensitivitySample3}, specificitySample3: {specificitySample3}"
    )
    print(
        f"sensitivitySample4: {sensitivitySample4}, specificitySample4: {specificitySample4}"
    )

def assignmentPart2():
    deltaSubSampling()
    perceptronSubSampling()


def main():
    #assignmentPart1()
    assignmentPart2()


if __name__ == "__main__":
    main()
