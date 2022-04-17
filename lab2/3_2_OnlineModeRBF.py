from calendar import EPOCH
from fileinput import filename

import matplotlib.pyplot as plt
import numpy as np

import dataGenerator
from rbf import RBF
import TwoLayerPerceptron

def point1():
    '''
    Compare the effect of the number of RBF units and their width for the two learning approaches. 
    Which error estimate should you choose as the criterion for these comparative analyses?
    '''
    sinData = dataGenerator.getSin(noise=True)
    xTrain, yTrain, xTest, yTest = sinData["xTrain"], sinData["yTrain"], sinData["xTest"], sinData["yTest"]  
    f = 'sin'
    # squareData = dataGenerator.getSquare(noise=True)
    # xTrain, yTrain, xTest, yTest = squareData["xTrain"], squareData["yTrain"], squareData["xTest"], squareData["yTest"]  
    # f = 'square'
    nRBF = np.arange(1,63, 1)

    nRBF = np.arange(1, 63, 1)

    learningRate = 0.1
    validationFraction = 0
    SIGMA = 1
    errors = []
    for n in nRBF:
        rbf = RBF(
            learningRate=learningRate,
            validationFraction=validationFraction,
            nRBF=n,
            sigma=SIGMA,
            learningType="sequential",
        )

        rbf.fit(xTrain=xTrain, yTrain=yTrain, epochs=30)
        yPred = rbf.predict(xTest)
        # yPred = np.sign(yPred) # sign function transformation for output on square(2x)
        err = rbf.getError(yReal=yTest, yPred=yPred)
        errors.append(err)

        print(f"At node {n}, err: {err}, sigma: {SIGMA}")
        if n % 5 == 0:
            plt.figure()
            plt.ylim(-1.2, 1.2)
            title = f'Predicted y-value VS Real y-value for {f}(2x)\n{n} Online RBFs and Sigma {SIGMA}'
            plt.title(title)
            plt.plot(xTrain, yTrain, label="Real")
            plt.plot(xTrain, yPred, label="Pred")
            plt.legend()
            plt.grid()
            filename = title.replace(" ", "_")
            filename = filename.replace("-", "")
            filename = filename.replace("\n", "")
            plt.savefig('images/OnlineModeRBF/point1/sin/'+ filename)

    filename2 = f'Residual Error vs Number of Online RBF for {f}(2X)\nSigma: {SIGMA}' 
    plt.figure()
    plt.title(filename2)
    plt.plot(nRBF, errors, label="Error per nRBFs")
    plt.xlabel("Number of RBF")
    plt.ylabel("Residual Error")
    plt.legend()
    plt.grid()
    filename2 = filename2.replace(" ", "_").replace("-", "").replace("\n", "")
    plt.savefig(f'images/OnlineModeRBF/point1/{f}/'+ filename2)


def point2():
    """
    What can you say about the rate of convergence and its dependence on the learning rate, 
    eta, for the on-line learning scheme?
    """
    etas = [
        0.1,
        0.01,
        0.001,
        0.0001,
        0.00001,
        0.000001,
        0.0000001,
        0.00000001,
        0.000000001,
    ]

    sinData = dataGenerator.getSin(noise=True)
    xTrain, yTrain, xTest, yTest = (
        sinData["xTrain"],
        sinData["yTrain"],
        sinData["xTest"],
        sinData["yTest"],
    )

    # squareData = dataGenerator.getSquare(noise=True)
    # xTrain, yTrain, xTest, yTest = squareData["xTrain"], squareData["yTrain"], squareData["xTest"], squareData["yTest"]

    nRBF = 30
    errors = []
    validationFraction = 0
    SIGMA = 1
    epochs = 200
    for eta in etas:
        learningRate = eta
        rbf = RBF(
            learningRate=eta,
            validationFraction=validationFraction,
            nRBF=nRBF,
            sigma=SIGMA,
            learningType="sequential",
        )

        rbf.fit(xTrain=xTrain, yTrain=yTrain, epochs=epochs)
        errors.append(rbf.epochErrors)

    # make plot now
    # x axis: noepochs
    # y axis: MSE error
    # show scatter?
    # show legend
    x = [i for i in range(1, epochs)]
    # print('len x: ', len(x))
    # print('len y: ', len(errors[0]))

    plt.figure()
    title = "Convergence for different learning rates"
    plt.title(title)
    plt.plot(x, errors[0], label=str(etas[0]))
    plt.plot(x, errors[1], label=str(etas[1]))
    plt.plot(x, errors[2], label=str(etas[2]))
    plt.plot(x, errors[3], label=str(etas[3]))
    plt.plot(x, errors[4], label=str(etas[4]))
    plt.plot(x, errors[5], label=str(etas[5]))
    plt.plot(x, errors[6], label=str(etas[6]))
    plt.plot(x, errors[7], label=str(etas[7]))
    plt.plot(x, errors[8], label=str(etas[8]))
    plt.xlabel("Number of epochs")
    plt.ylabel("Residual Error")
    plt.legend()
    plt.grid()
    plt.show()


def point3(data):
    '''
    What are the main effects of changing the width of RBFs?
    With other words, change sigma
    '''
    if data == 'sin':
        sinData = dataGenerator.getSin(noise=True)
        xTrain, yTrain, xTest, yTest = sinData["xTrain"], sinData["yTrain"], sinData["xTest"], sinData["yTest"] 
    elif data == 'square': 
        squareData = dataGenerator.getSquare(noise=True)
        xTrain, yTrain, xTest, yTest = squareData["xTrain"], squareData["yTrain"], squareData["xTest"], squareData["yTest"]  
    else:
        raise Exception(f'\"{data}\" is not a valid data type, please try \"sin\" or \"square\"') 
    
    SIGMA = [0.5, 1, 1.2, 1.4, 1.6, 1.8] 

    nRBF = 24
    epochs=45
    learningRate=0.2
    validationFraction=0
    errors = []
    for sigma in SIGMA:
        rbf = RBF( learningRate=learningRate, validationFraction=validationFraction, 
            nRBF=nRBF, 
            sigma=sigma,
            learningType='sequential',
        )
        rbf.fit(xTrain=xTrain, yTrain=yTrain, epochs=epochs)
        errors.append(rbf.epochErrors)

    x = [i for i in range(1,epochs)]
    filename3 = f'Convergence on {data}(2x) for different widths (sigma)' 
    plt.figure()
    plt.title(filename3)
    for ie, e in enumerate(errors):
        plt.plot(x, e, label= 'sigma: '+str(SIGMA[ie]))
    # plt.plot(x, errors[1], label= 'sigma: '+str(SIGMA[1]))
    # plt.plot(x, errors[2], label= 'sigma: '+str(SIGMA[2]))
    # plt.plot(x, errors[3], label= 'sigma: '+str(SIGMA[3]))
    # plt.plot(x, errors[4], label= 'sigma: '+str(SIGMA[4]))
    plt.xlabel('Number of epochs')
    plt.ylabel('Residual Error')
    plt.legend()
    plt.grid()
    filename3 = filename3.replace(" ", "_").replace("-", "").replace("\n", "")
    plt.savefig('images/OnlineModeRBF/sin/point3/'+ filename3)


def point4():
    """
    How important is the positioning of the RBF nodes in the input space?
    What strategy did you choose? Is it better than random positioning of the
    RBF nodes? Please support your conclusions with quantitative evidence
    (e.g., error comparison).
    """
    """ 
    Compare different ways of positioning the nodes
    """

    sinData = dataGenerator.getSin(noise=True)
    xTrain, yTrain, xTest, yTest = (
        sinData["xTrain"],
        sinData["yTrain"],
        sinData["xTest"],
        sinData["yTest"],
    )

    # squareData = dataGenerator.getSquare(noise=True)
    # xTrain, yTrain, xTest, yTest = squareData["xTrain"], squareData["yTrain"], squareData["xTest"], squareData["yTest"]

    nRBF = 30
    errors = []
    validationFraction = 0
    SIGMA = 1
    epochs = 200
    learningRate = 0.01
    placementTypes = ["choice", "random"]

    for placementType in placementTypes:
        rbf = RBF(
            learningRate=learningRate,
            validationFraction=validationFraction,
            nRBF=nRBF,
            sigma=SIGMA,
            learningType="sequential",
            positionType=placementType,
        )
        rbf.fit(xTrain=xTrain, yTrain=yTrain, epochs=epochs)
        errors.append(rbf.epochErrors)

    # make plot now
    # x axis: noepochs
    # y axis: MSE error
    # show scatter?
    # show legend
    x = [i for i in range(1, epochs)]
    # print('len x: ', len(x))
    # print('len y: ', len(errors[0]))

    plt.figure()
    title = "Convergence for different types of RBF center placements"
    plt.title(title)
    plt.plot(x, errors[0], label=placementTypes[0])
    plt.plot(x, errors[1], label=placementTypes[1])
    plt.xlabel("Number of epochs")
    plt.ylabel("Residual Error")
    plt.legend()
    plt.grid()
    plt.show()


def point5():
    '''
    Also, for the same network models estimate their test performance on the
    original clean data used in section 3.1 (a corresponding test subset but
    without noise) and compare your findings.
    '''
    sinData = dataGenerator.getSin(noise=False)
    xTrain, yTrain, xTest, yTest = sinData["xTrain"], sinData["yTrain"], sinData["xTest"], sinData["yTest"]  
    f = 'sin'
    # squareData = dataGenerator.getSquare(noise=True)
    # xTrain, yTrain, xTest, yTest = squareData["xTrain"], squareData["yTrain"], squareData["xTest"], squareData["yTest"]  
    # f = 'square'
    nRBF = np.arange(1,63, 1)

    learningRate=0.1
    validationFraction=0
    SIGMA = 1
    errors = []
    for n in nRBF:
        rbf = RBF( learningRate=learningRate, validationFraction=validationFraction, 
            nRBF=n, 
            sigma=SIGMA,
            learningType='sequential',
        )

        rbf.fit(xTrain=xTrain, yTrain=yTrain, epochs=30)
        yPred = rbf.predict(xTest)
        # yPred = np.sign(yPred) # sign function transformation for output on square(2x)
        err = rbf.getError(yReal=yTest, yPred=yPred)
        errors.append(err)

        print(f'At node {n}, err: {err}, sigma: {SIGMA}')
        if n%5==0:
            plt.figure()
            plt.ylim(-1.2, 1.2)
            title = f'Predicted y-value VS Real y-value for {f}(2x)\n{n} Online RBFs and Sigma {SIGMA}'
            plt.title(title)
            plt.plot(xTrain, yTrain, label='Real')
            plt.plot(xTrain, yPred, label='Pred')
            plt.legend()
            plt.grid()
            filename = title.replace(" ", "_")
            filename = filename.replace("-", "")
            filename = filename.replace("\n", "")
            plt.savefig('images/OnlineModeRBF/point5/sin/'+ filename)

    filename2 = f'Residual Error vs Number of Online RBF for {f}(2X)\nSigma: {SIGMA}' 
    plt.figure()
    plt.title(filename2)
    plt.plot(nRBF, errors, label='Error per nRBFs')
    plt.xlabel('Number of RBF')
    plt.ylabel('Residual Error')
    plt.legend()
    plt.grid()
    filename2 = filename2.replace(" ", "_").replace("-", "").replace("\n", "")
    plt.savefig(f'images/OnlineModeRBF/point5/{f}/'+ filename2)

    pass


def point6(data='sin'):
    """
    Please compare your optimal RBF network trained in batch mode with
    a single-hidden-layer perceptron trained with backprop (also in batch
    mode), which you implemented in the first lab assignment. Please use
    the same number of hidden units as in the RBF network. The comparison
    should be made for both functions: sin(2x) and square(2x), only for the
    noisy case. Please remember that generalisation performance and training
    time are of greatest interest.
    """
    ''' Generate function data '''
    # minX = -5
    # maxX = 5
    # stepSize = 0.5
    # numSteps = int((np.abs(minX) + np.abs(maxX)) / 0.5) 
    # x = np.linspace(minX, maxX, numSteps)
    # y = np.linspace(minX, maxX, numSteps)
    # x3D = np.outer(x, np.ones(numSteps))
    # y3D = x3D.copy().T
    # z = np.exp(-(x3D**2 + y3D**2)*.1) - 0.5
    
    # 3-D plotting code inspired by: https://www.geeksforgeeks.org/three-dimensional-plotting-in-python-using-matplotlib/
    """
    Shapes
    xx = (20, 20)
    yy = (20, 20)
    x = (20,)
    y = (20,)
    z = (20, 20)
    """
    # nData = sum(x.shape) * sum(y.shape)
    # targets = np.reshape(z, (1, nData))
    # xx, yy = np.meshgrid(x, y)
    # patterns = np.vstack([np.reshape(xx, (1, nData)), np.reshape(yy, (1, nData))]) 
    ''' Train the network and visualise the approximated function '''
    if (data=='sin'):
        sinData = dataGenerator.getSin(noise=True)
        xTrain, yTrain, xTest, yTest = (
            sinData["xTrain"].T,
            sinData["yTrain"].T,
            sinData["xTest"].T,
            sinData["yTest"].T,
        )
    else:
        squareData = dataGenerator.getSquare(noise=True)
        xTrain, yTrain, xTest, yTest = (
            squareData["xTrain"].T,
            squareData["yTrain"].T,
            squareData["xTest"].T,
            squareData["yTest"].T,
        )
    LEARNING_RATE = 0.01
    ALPHA = 0.8
    MSEs = []
    EPOCHS = 500
    nHIDDEN = 24
    # patterns: (2, 400)
    # targets:  (1, 400)
    # out:      (1, 400)
    perceptron = TwoLayerPerceptron.TwoLayerPerceptron(
        xInput=xTrain, 
        targets=yTrain, 
        numHidden=nHIDDEN, 
        numOutput=2, 
        learningRate=LEARNING_RATE, 
        alpha=ALPHA,
    )    
    _, mse, _, _ = perceptron.train(EPOCHS, [])
    _, _, out, _ = perceptron.forwardPass()
    print('out: ', out.shape)
    plt.figure()
    title = f'Approximation of {data}(2x) with Two-layer Perceptron\nepochs: {EPOCHS}, hidden nodes: {nHIDDEN}, MSE: {mse[-1]:.2f}'
    plt.title(title)
    plt.ylabel(f'{data}(2x)')
    plt.xlabel('x')
    plt.plot(np.ravel(xTest), np.ravel(yTest), label='Real')
    plt.plot(np.ravel(xTest), np.ravel(out), label='Pred')
    plt.legend()
    plt.grid()
    filename = title.replace(' ', '_')
    filename = filename.replace(',', '_')
    filename = filename.replace('\n', '_')
    filename = filename.replace(':', '_')
    filename = filename.replace('.', '_')
    
    plt.savefig(f'images/OnlineModeRBF/point5/' + filename)

    # plotFunction3D(out, x3D, y3D, nData, 24, None)
    pass
    # if (data=='sin'):
    #     sinData = dataGenerator.getSin(noise=True)
    #     xTrain, yTrain, xTest, yTest = (
    #         sinData["xTrain"].T,
    #         sinData["yTrain"].T,
    #         sinData["xTest"].T,
    #         sinData["yTest"].T,
    #     )
    # else:
    #     squareData = dataGenerator.getSquare(noise=True)
    #     xTrain, yTrain, xTest, yTest = (
    #         squareData["xTrain"].T,
    #         squareData["yTrain"].T,
    #         squareData["xTest"].T,
    #         squareData["yTest"].T,
    #     )

    # EPOCHS = 500
    # nHIDDEN = 24
    # TLP = TwoLayerPerceptron.TwoLayerPerceptron(
    #     xInput=xTrain, 
    #     targets=yTrain, 
    #     numHidden=nHIDDEN, 
    #     numOutput=2, 
    #     learningRate=0.01, 
    #     alpha=0.8,
    # )    
    # _, mse, _, _ = TLP.train(epochs=EPOCHS)
    # _, _, yPred, _ = TLP.forwardPass()
    # print(np.ravel(yPred))
    # print(mse[-1])
    # plt.figure()
    # title = f'Approximation of {data}(2x) with Two-layer Perceptron\nepochs: {nHIDDEN}, hidden nodes: {nHIDDEN}, MSE: {mse[-1]}'
    # plt.ylabel(f'{data}(2x)')
    # plt.xlabel('x')
    # plt.plot(np.ravel(xTest), np.ravel(yTest), label='Real')
    # plt.plot(np.ravel(xTest), np.ravel(yPred), label='Pred')
    # plt.legend()
    # plt.grid()
    # plt.show()


def plotFunction3D(out, x, y, nData, nHidden, directory, valSize=None):
    gridSize = int(np.sqrt(nData))
    zz = np.reshape(out, (gridSize, gridSize))
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    # syntax for plotting
    titleString = f"Approximated Function, hidden layers: {nHidden}"
    if valSize != None:
        titleString += f", validation split: {valSize}"
    ax.plot_surface(x, y, zz, cmap ='viridis', edgecolor ='green')
    ax.set_title(titleString)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.grid()
    filename = titleString
    filename = filename.replace(": ", "_")
    filename = filename.replace(", ", "_")
    filename = filename.replace(" ", "_")
    filename = filename.replace(".", "")
    

if __name__=='__main__':
    #point1()
    #point2()
    #point3(data='square')
    # point4()
    point6(data='sin')

