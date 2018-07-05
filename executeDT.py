from beautifultable import BeautifulTable
import matplotlib.pyplot as plt
import sys
from trainDT import readInput, DataPoint, printTree, createListForPlotting


def setLeftAndRight(objectList):
    """
    Set the left and right children for each object in the given list
    :param objectList: the list of objects whose children need to be set
    :return: the updated list of objects
    """
    for index in range(len(objectList)):
        if objectList[index].id == 0:
            for index2 in range(len(objectList)):
                if objectList[index2].id == 1:
                    objectList[index].leftNode = objectList[index2]
            for index3 in range(len(objectList)):
                if objectList[index3].id == 12:
                    objectList[index].rightNode = objectList[index3]

        elif objectList[index].id == 1:
            for index2 in range(len(objectList)):
                if objectList[index2].id == 2:
                    objectList[index].leftNode = objectList[index2]
            for index3 in range(len(objectList)):
                if objectList[index3].id == 7:
                    objectList[index].rightNode = objectList[index3]

        elif objectList[index].id == 2:
            for index2 in range(len(objectList)):
                if objectList[index2].id == 3:
                    objectList[index].leftNode = objectList[index2]
                    objectList[index2].isLeafNode = True
            for index3 in range(len(objectList)):
                if objectList[index3].id == 4:
                    objectList[index].rightNode = objectList[index3]

        elif objectList[index].id == 4:
            for index2 in range(len(objectList)):
                if objectList[index2].id == 5:
                    objectList[index].leftNode = objectList[index2]
                    objectList[index2].isLeafNode = True
            for index3 in range(len(objectList)):
                if objectList[index3].id == 6:
                    objectList[index].rightNode = objectList[index3]
                    objectList[index3].isLeafNode = True

        elif objectList[index].id == 7:
            for index2 in range(len(objectList)):
                if objectList[index2].id == 8:
                    objectList[index].leftNode = objectList[index2]
                    objectList[index2].isLeafNode = True
            for index3 in range(len(objectList)):
                if objectList[index3].id == 9:
                    objectList[index].rightNode = objectList[index3]

        elif objectList[index].id == 9:
            for index2 in range(len(objectList)):
                if objectList[index2].id == 10:
                    objectList[index].leftNode = objectList[index2]
                    objectList[index2].isLeafNode = True
            for index3 in range(len(objectList)):
                if objectList[index3].id == 11:
                    objectList[index].rightNode = objectList[index3]
                    objectList[index3].isLeafNode = True

        elif objectList[index].id == 12:
            for index2 in range(len(objectList)):
                if objectList[index2].id == 13:
                    objectList[index].leftNode = objectList[index2]
            for index3 in range(len(objectList)):
                if objectList[index3].id == 18:
                    objectList[index].rightNode = objectList[index3]
                    objectList[index3].isLeafNode = True

        elif objectList[index].id == 13:
            for index2 in range(len(objectList)):
                if objectList[index2].id == 14:
                    objectList[index].leftNode = objectList[index2]
                    objectList[index2].isLeafNode = True
            for index3 in range(len(objectList)):
                if objectList[index3].id == 15:
                    objectList[index].rightNode = objectList[index3]

        elif objectList[index].id == 15:
            for index2 in range(len(objectList)):
                if objectList[index2].id == 16:
                    objectList[index].leftNode = objectList[index2]
                    objectList[index2].isLeafNode = True
            for index3 in range(len(objectList)):
                if objectList[index3].id == 17:
                    objectList[index].rightNode = objectList[index3]
                    objectList[index3].isLeafNode = True
        else:
            objectList[index].rightNode = objectList[index].leftNode = None

    return objectList


def createTreeList(dt_file):
    """
    Creates a list of tree objects by reading in the decision tree file created by the trainDT.py
    :param dt_file: the file created by trainDT.py, which contains the decision tree object
    :return: the list of tree objects
    """
    f = open(dt_file, 'r')
    data = f.readlines()
    treeList = []
    for line in data:
        treeList.append(line.strip().split(','))
    return treeList


def createObjects(treeList):
    """
    For a given list, create an object and set all its features
    :param treeList: the list containing the tree elements in the form of a list
    :return: the list of objects for the tree
    """
    dataPointList = []
    for index in range(len(treeList)):
        if treeList[index][2] == '0':
            dataPoint = DataPoint(index, 0, 0, 0)
            dataPoint.attribNumber = float(treeList[index][0])
            dataPoint.threshold = float(treeList[index][1])
            dataPoint.leafValue = int(treeList[index][2])
            dataPointList.append(dataPoint)
        else:
            dataPoint = DataPoint(index, 0, 0, 0)
            dataPoint.leafValue = int(treeList[index][2])
            dataPoint.isLeafNode = True
            dataPointList.append(dataPoint)
    return dataPointList


def classifySamples(sample, root):
    """
    Given the sample, classify it into one of the 4 classes
    :param sample: the sample to be classified
    :param root: the root of the decision tree
    :return: the leaf value containing the class
    """
    # check which attribute to split on and based on that recursively call
    if root.attribNumber == 1:
        if sample.x > root.threshold:
            return classifySamples(sample, root.rightNode)
        else:
            return classifySamples(sample, root.leftNode)
    elif root.attribNumber == 2:
        if sample.y > root.threshold:
            return classifySamples(sample, root.rightNode)
        else:
            return classifySamples(sample, root.leftNode)
    else:
        value = root.leafValue
        return value


def classifyForPlot(sample, root):
    """
    Classify given sample for the decision plot
    :param sample: the sample to classify
    :param root: the root of the tree
    :return: the leaf value
    """
    if root.attribNumber == 1:
        if sample[0] > root.threshold:
            return classifyForPlot(sample, root.rightNode)
        else:
            return classifyForPlot(sample, root.leftNode)
    elif root.attribNumber == 2:
        if sample[1] > root.threshold:
            return classifyForPlot(sample, root.rightNode)
        else:
            return classifyForPlot(sample, root.leftNode)
    else:
        value = root.leafValue
        return value


def classifyFull(sampleList, pointList, root):
    """
    Classify the list of points and plot them on the decision plot
    :param sampleList: the list of points to draw the region
    :param pointList: the list of testing data points
    :param root: the root of the decision tree
    :return:
    """
    correct1x, correct1y, correct2x, correct2y = [], [], [], []
    correct3x, correct3y, correct4x, correct4y = [], [], [], []
    for index in range(len(sampleList)):
        prediction = classifyForPlot(sampleList[index], root)
        if prediction == 1:
            correct1x.append(sampleList[index][0])
            correct1y.append(sampleList[index][1])

        elif prediction == 2:
            correct2x.append(sampleList[index][0])
            correct2y.append(sampleList[index][1])

        elif prediction == 3:
            correct3x.append(sampleList[index][0])
            correct3y.append(sampleList[index][1])

        elif prediction == 4:
            correct4x.append(sampleList[index][0])
            correct4y.append(sampleList[index][1])

    plt.plot(correct1x, correct1y, 'green')
    plt.plot(correct2x, correct2y, 'blue')
    plt.plot(correct3x, correct3y, 'red')
    plt.plot(correct4x, correct4y, 'black')

    point1x, point1y, point2x, point2y, point3x, point3y, point4x, point4y = [], [], [], [], [], [], [], []
    for index in range(len(pointList)):
       if pointList[index].z == 1:
           point1x.append(pointList[index].x)
           point1y.append(pointList[index].y)
       elif pointList[index].z == 2:
           point2x.append(pointList[index].x)
           point2y.append(pointList[index].y)
       elif pointList[index].z == 3:
           point3x.append(pointList[index].x)
           point3y.append(pointList[index].y)
       else:
           point4x.append(pointList[index].x)
           point4y.append(pointList[index].y)

    plt.plot(point1x, point1y, 'c*', label='Bolt')
    plt.plot(point2x, point2y, 'y^', label='Nut')
    plt.plot(point3x, point3y, 'mx', label='Ring')
    plt.plot(point4x, point4y, 'w+', label='Scrap')
    plt.title("Decision boundary")
    plt.xlabel("Six fold rotational symmetry")
    plt.ylabel("Eccentricity")
    plt.legend(loc='upper right')
    #plt.show()


def classifyAllSamples(sampleList, root):
    """
    Function to classify the given list of samples using the classifier created above
    :param sampleList: the list of samples
    :param root: the root node of the tree
    :return: the accuracy of the classifier and its mean per class accuracy
    """
    correct, incorrect = 0, 0
    meanPerClass = 0
    # create a 2D matrix of 4x4 with all 0 elements
    correctClass, incorrectClass = [0 for _ in range(4)], [0 for _ in range(4)]

    for index in range(len(sampleList)):
        classValue = classifySamples(sampleList[index], root)  # get the predicted class of a given sample
        if classValue == sampleList[index].z:  # check if the predicted class is the same as actual class
            correct += 1
            # set the count of correctly classified instance in the list at its corresponding location
            for idx in range(4):
                if sampleList[index].z == idx + 1:
                    correctClass[idx] += 1
        else:
            incorrect += 1
            # set the count of incorrectly classified instance in the list at its corresponding location
            for idx in range(4):
                if sampleList[index].z == idx + 1:
                    incorrectClass[idx] += 1

    accuracy = correct / (correct + incorrect) * 100    
    # compute the mean per class accuracy
    for idx in range(4):
        meanPerClass += correctClass[idx] / (correctClass[idx] + incorrectClass[idx])
    meanPerClass /= 4

    return meanPerClass, accuracy


def buildConfusionMatrix(sampleList, root):
    """
    Function to build the confusion matrix
    :param sampleList: the list of all samples
    :param root: the root node of the tree
    :return: the confusion matrix
    """
    # create a 2D matrix of 4x4 size with all 0 elements
    confusionMatrix = [[0 for _ in range(4)] for _ in range(4)]                
    # iterate over the entire list of samples to check whether the class is correctly predicted or not
    for index in range(len(sampleList)):
        classValue = classifySamples(sampleList[index], root)
        actualClassValue = sampleList[index].z
        confusionMatrix[int(classValue) - 1][int(actualClassValue) - 1] += 1

    return confusionMatrix


def printTable(confusionMatrix):
    """
    Helper function that is used to print the confusion matrix
    :param confusionMatrix: list of lists containing the confusion matrix
    :return: None
    """
    table = BeautifulTable()
    classes = ["Bolt", "Nut", "Ring", "Scrap"]
    table.column_headers = classes
    for line in confusionMatrix:
        table.append_row(line)
    table.insert_column(0, "Predicted (down) \ Actual (across)", classes)
    print(table)


def calculateProfit(confusionMatrix):
    """
    Compute the total profit obtained by using the classifier
    :param confusionMatrix: the confusion matrix to compute profit
    :return: the total profit
    """
    profitMatrix = [[20, -7, -7, -7], [-7, 15, -7, -7], [-7, -7, 5, -7], [-3, -3, -3, -3]]     # the given profit matrix        
    totalProfit = 0
    for outerIndex in range(len(profitMatrix)):
        for innerIndex in range(len(profitMatrix)):
            # compute profit obtained for each class
            totalProfit += profitMatrix[outerIndex][innerIndex] * confusionMatrix[outerIndex][innerIndex]    

    return totalProfit


def main():
    """
    The main function
    :return: None
    """
    if len(sys.argv) != 3:
        print("USAGE: python3 executeDT <DT_file> <data_file>")
        print("<DT_file> = decision tree file created by trainDT.py")
        print("<data_file> = testing data file")

    filename = sys.argv[1]
    pointList = readInput(sys.argv[2])

    print('------------- Before pruning --------------')

    treeList = createTreeList(filename)
    objectList = createObjects(treeList)
    new_objectList = setLeftAndRight(objectList)
    root_object = new_objectList[0]
    confusionMatrix = buildConfusionMatrix(pointList, root_object)
    printTable(confusionMatrix)
    data = createListForPlotting()
    classifyFull(data, pointList, root_object)
    print()
    print('********** Recognition rate ***********')
    mpc_accuracy, accuracy = classifyAllSamples(pointList, root_object)
    print('Mean per class accuracy is: ', str(mpc_accuracy))
    print('Accuracy is: ', str(accuracy))
    print('***************************************')
    print()
    print('************ Profit ***********')
    print('Total profit computed is:', str(calculateProfit(confusionMatrix)))
    print('*******************************')

    print('----------- After pruning ------------')

    treeList_pruned = createTreeList(filename)
    objectList_pruned = createObjects(treeList_pruned)
    new_objectList_pruned = setLeftAndRight(objectList_pruned)
    root_object_pruned = new_objectList_pruned[0]
    confusionMatrix = buildConfusionMatrix(pointList, root_object_pruned)

    printTable(confusionMatrix)
    data_pruned = createListForPlotting()
    classifyFull(data_pruned, pointList, root_object)
    print()
    print('********** Recognition rate ***********')
    mpc_accuracy, accuracy = classifyAllSamples(pointList, root_object)
    print('Mean per class accuracy is: ', str(mpc_accuracy))
    print('Accuracy is: ', str(accuracy))
    print('***************************************')
    print()
    print('************ Profit ***********')
    print('Total profit computed is:', str(calculateProfit(confusionMatrix)))
    print('*******************************')


if __name__ == '__main__':
    main()
