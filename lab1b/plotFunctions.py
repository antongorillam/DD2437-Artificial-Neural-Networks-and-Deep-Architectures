import matplotlib.pyplot as plt
import numpy as np

def plotMetric(metricList, nHidden, directory, metric, alpha, learningRate):
    #errorsA, learningTypeA, errorsB, learningTypeB, learningRate, withBias, directory
    titleString = f"{metric} of Two-layer Perceptron with\nhidden nodes: {nHidden}, alpha: {alpha}, learning rate {learningRate}."
    plt.figure(dpi=100) #figsize=(15, 10), dpi=100)
    plt.plot(metricList, label="Misclassifications")
    plt.title(titleString)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    # plt.ylim([0,1.2])
    plt.legend()
    plt.grid()
    filename = titleString
    filename = filename.replace(": ", "_")
    filename = filename.replace(", ", "_")
    filename = filename.replace(" ", "_")
    filename = filename.replace(".", "")
    filename = filename.replace("\n", " ")
    plt.savefig(directory + filename)

def plotInterpolate(ValMSEs, TrainMSEs,validationSplit, directory, metric, alpha, learningRate):
    #errorsA, learningTypeA, errorsB, learningTypeB, learningRate, withBias, directory
    titleString = f"Interpolated {metric} of Two-layer Perceptrons for different validation-set sizes"
    plt.figure(dpi=100) #figsize=(15, 10), dpi=100)
    plt.title(titleString)
    plt.xlabel("Validation-set size")
    plt.ylabel(metric)
    # yLim = np.max(MSEs) + 0.05

    # plt.ylim([0, yLim])
    # for mse, size in zip(MSEs, validationSplit):
    plt.plot(validationSplit, ValMSEs, label='MSE on validation-set')
    plt.plot(validationSplit, TrainMSEs, label='MSE on training-set')
    
    plt.legend()
    plt.grid()
    filename = titleString
    filename = filename.replace(": ", "_")
    filename = filename.replace(", ", "_")
    filename = filename.replace(" ", "_")
    filename = filename.replace(".", "")
    filename = filename.replace("\n", " ")
    # plt.savefig(directory + filename)
    plt.show()

def plotMSEOnHiddenNodes(MSEs, nHidden, directory, metric, alpha, learningRate):
    #errorsA, learningTypeA, errorsB, learningTypeB, learningRate, withBias, directory
    titleString = f"{metric} of Two-layer Perceptrons with\nalpha: {alpha}, learning rate {learningRate}"
    plt.figure(dpi=100) #figsize=(15, 10), dpi=100)
    plt.title(titleString)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    yLim = np.max(MSEs) + 0.05
    plt.ylim([0, yLim])
    for mse, label in zip(MSEs, nHidden):
        plt.plot(mse, label=f'{label} hidden nodes')
    
    plt.legend()
    plt.grid()
    filename = titleString
    filename = filename.replace(": ", "_")
    filename = filename.replace(", ", "_")
    filename = filename.replace(" ", "_")
    filename = filename.replace(".", "")
    filename = filename.replace("\n", " ")
    # plt.savefig(directory + filename)
    plt.show()


def plotTrainingAndValidation(trainingList ,validationList, nHidden, directory, metricString, alpha, learningRate, validationSize=None):
    #errorsA, learningTypeA, errorsB, learningTypeB, learningRate, withBias, directory
    titleString = f"{metricString} of Two-layer Perceptron with \nhidden nodes: {nHidden}, alpha: {alpha}, learning rate {learningRate}"
    if validationSize != None:
        titleString += f", validationSize: {validationSize}"

    plt.figure() #figsize=(15, 10), dpi=100)
    plt.plot(trainingList, label="Training")
    plt.plot(validationList, label="Validation")
    plt.title(titleString)
    plt.xlabel("Epochs")
    plt.ylabel(metricString)
    plt.ylim([0, 0.35])
    plt.legend()
    plt.grid()
    filename = titleString
    filename = filename.replace(": ", "_")
    filename = filename.replace(", ", "_")
    filename = filename.replace(" ", "_")
    filename = filename.replace(".", "")
    filename = filename.replace("\n", " ")
    # plt.savefig(directory + filename)
    plt.show()

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
    # plt.show()
    plt.savefig(directory + filename)

def plotFunction3DNew(out, x, y, nData, nHidden, directory, valSize=None):
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
    plt.savefig(directory + filename)

        

if __name__ == "__main__":
    misclassification = [100, 99, 75, 55, 54, 63, 73, 80, 82, 81, 76, 72, 70, 69, 69, 67, 67, 66, 64, 63, 62, 62, 62, 62, 62, 62, 61, 61, 59, 59, 59, 58, 57, 57, 57, 57, 57, 57, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57]
    mse = [1.0384144224430987, 1.0270891740174501, 1.013507099869016, 1.0026856341999153, 0.9958699031600836, 0.9921485451675247, 0.9901105616943082, 0.9886935966919662, 0.9873294737496798, 0.9857973062439833, 0.9840504187283118, 0.9821040689757418, 0.9799827247072126, 0.977703141248473, 0.9752731651310156, 0.9726950344450608, 0.9699686320428768, 0.9670936245863077, 0.9640706809654804, 0.9609021577672715, 0.9575924995074485, 0.9541484509002283, 0.9505791010872838, 0.9468957623966895, 0.9431116967681844, 0.9392417185215494, 0.9353017123281566, 0.9313081079538371, 0.927277350057976, 0.9232253943811712, 0.9191672530316359, 0.9151166026860527, 0.9110854612115816, 0.9070839309503561, 0.9031200008683151, 0.8991993949523924, 0.8953254505143191, 0.8914990072121572, 0.8877182853635188, 0.883978730196392, 0.8802727967535446, 0.8765896479418208, 0.8729147354659458, 0.8692292300065948, 0.8655092631456643, 0.8617249398209041, 0.8578390779345241, 0.8538056339567718, 0.8495677850070648, 0.8450556674048888]
    
    plotMetric(metricList=misclassification, learningRate=0.001, nHidden=3, directory="images/3.1.1_point_1/", metric = "Misclassification")
    plotMetric(metricList=mse, learningRate=0.001, nHidden=3, directory="images/3.1.1_point_1/", metric = "Mean Squared Error")

