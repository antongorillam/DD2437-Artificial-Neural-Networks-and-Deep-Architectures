from fileinput import filename
from rbf import RBF
import matplotlib.pyplot as plt
import numpy as np
import dataGenerator

# sätt 1 från 09, sätt 1 från 10

nRBF = [i for i in range(1, 63)]
errors = []

sinData = dataGenerator.getSin()
xTrain, yTrain, xTest, yTest = sinData["xTrain"], sinData["yTrain"], sinData["xTest"], sinData["yTest"]  
print('xtrain ', xTrain)

# squareData = dataGenerator.getSquare()
# xTrain, yTrain, xTest, yTest = squareData["xTrain"], squareData["yTrain"], squareData["xTest"], squareData["yTest"]  
#print('xtrain ', xTrain)


learningRate=0.5
validationFraction=0
SIGMA=1

    
for n in nRBF:
    rbf = RBF(
        learningRate=learningRate,
        validationFraction=validationFraction,
        nRBF=n,
        sigma=SIGMA,
    )
    rbf.fit(xTrain=xTrain, yTrain=yTrain)
    yPred = rbf.predict(xTest)
    # yPred = np.sign(yPred) # sign function transformation for output on square(2x)
    err = rbf.getError(yReal=yTest, yPred=yPred)
    print(f'At node {n}, err: {err}')
    if n%5==0:
        plt.figure()
        
        title = f'Predicted y-value VS Real y-value for un-signed square(2x)\n{n} RBFs'
        plt.title(title)
        plt.plot(xTrain, yTrain, label='Real')
        plt.plot(xTrain, yPred, label='Pred')
        plt.legend()
        plt.grid()
        filename = title.replace(" ", "_")
        filename = filename.replace("-", "")
        filename = filename.replace("\n", "")
        plt.savefig('images/BatchModeRBF/square'+ filename)
        # plt.show()
        
    errors.append(err)

plt.figure()
plt.title('Residual Error vs Number of RBF for square(2X)')
plt.plot(nRBF, errors, label='Error per nRBFs')
plt.xlabel('Number of RBF')
plt.ylabel('Residual Error')
plt.legend()
plt.grid()
plt.show()
