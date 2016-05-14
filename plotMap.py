import matplotlib.pyplot as plt
import numpy as np

def plotMap(dataSet):
    dataSet = np.array(dataSet)
    plt.scatter(np.array(filter(lambda value: value[2] == 1, dataSet))[:, 0],
                np.array(filter(lambda value: value[2] == 1, dataSet))[:, 1],
                color="red", s=30)
    plt.scatter(np.array(filter(lambda value: value[2] == 2, dataSet))[:, 0],
                np.array(filter(lambda value: value[2] == 2, dataSet))[:, 1],
                color="green", s=30)
    plt.scatter(np.array(filter(lambda value: value[2] == 3, dataSet))[:, 0],
                np.array(filter(lambda value: value[2] == 3, dataSet))[:, 1],
                color="blue", s=30)
    plt.scatter(np.array(filter(lambda value: value[2] == 4, dataSet))[:, 0],
                np.array(filter(lambda value: value[2] == 4, dataSet))[:, 1],
                color="purple", s=30)
    #map(lambda k: k.reshape(len(k), 1)[1], X.reshape(len(X), 3))
    # plt.scatter(np.array(dataSet)[0:20, 0], np.array(dataSet)[0:20, 1], color="red", s=30)
    # plt.scatter(np.array(dataSet)[20:, 0], np.array(dataSet)[20:, 1], color="red", s=30)
    plt.plot([np.array(dataSet)[:, 0].min() - 5, 0, np.array(dataSet)[:, 0].max() + 5], [0, 0, 0], color='green')
    plt.plot([0, 0, 0], [np.array(dataSet)[:, 1].min() - 5, 0, np.array(dataSet)[:, 1].max() + 5], color='green')
    plt.show()

def plotPartone(dataSet, color):
    dataSet = np.array(dataSet)
    plt.scatter(np.array(dataSet)[:, 0], np.array(dataSet)[:, 1], color=color, s=30)


def plotPart2(leftData, rightData):
    plotPartone(leftData, "red")
    plotPartone(rightData, "green")
    plt.plot([np.array(np.vstack([leftData, rightData]))[:, 0].min() - 5, 0, np.array(np.vstack([leftData, rightData]))[:, 0].max() + 5], [0, 0, 0], color='green')
    plt.plot([0, 0, 0], [np.array(np.vstack([leftData, rightData]))[:, 1].min() - 5, 0, np.array(np.vstack([leftData, rightData]))[:, 1].max() + 5], color='green')
    plt.show()

def getClassLabel(dataSet):
    classLabel = []
    for i in range(len(dataSet)):
        classLabel.append(dataSet[i, -1])
    classLabel = list(set(classLabel))
    return classLabel

def plotTree(dataSet, myTree):
    plt.scatter(np.array(dataSet)[:, 0], np.array(dataSet)[:, 1], color="red", s=30)
    plt.plot([np.array(dataSet)[:, 0].min() - 5, 0, np.array(dataSet)[:, 0].max() + 5], [0, 0, 0], color='green')
    plt.plot([0, 0, 0], [np.array(dataSet)[:, 1].min() - 5, 0, np.array(dataSet)[:, 1].max() + 5], color='green')


    plt.show()

