import math
import operator
import matplotlib.pyplot as plt
import sys


class DataPoint:
    """
    Class of each data point in the given csv file, with 3 attributes
    """
    def __init__(self, id, attrib1, attrib2, classValue):
        self.id = id
        self.x = attrib1
        self.y = attrib2
        self.z = classValue
        self.threshold = 0
        self.countClass = [0, 0, 0, 0]
        self.rightList = []
        self.leftList = []
        self.attribNumber = 0
        self.rightNode = None
        self.leftNode = None
        self.isLeafNode = False
        self.leafValue = 0


def readInput(filename):
    """
    Reads the file and creates a list of objects
    :param filename: the file to be read
    :return: the list of objects
    """
    file = open(filename, 'r')
    data = file.readlines()
    pointList = []
    for line in range(len(data)):
        store = data[line].strip().split(',')
        # check for empty spaces or new line characters
        if len(store) != 1:
            dataPoint = DataPoint(line, float(store[0]), float(store[1]), int(store[2]))
            pointList.append(dataPoint)
    return pointList


def countClass(pointList):
    """
    Compute the count of each class in a given sample
    :param pointList: the list of samples
    :return: the list of counts of each class
    """
    allCounts = [0, 0, 0, 0]
    for index in range(len(pointList)):
        allCounts[pointList[index].z - 1] += 1
    return allCounts


def findEntropyChild(totalCount, child):
    """
    Compute the entropy of given subtree
    :param totalCount: the total count of objects in the subtree
    :param child: the side of subtree, i.e., right or left
    :return: the computed entropy
    """
    compute = 0
    allCounts = [0, 0, 0, 0]
    # iterate over all objects in the subtree and count number of classes
    for element in range(len(child)):
        value = child[element].z
        if value == 1:
            allCounts[0] += 1
        elif value == 2:
            allCounts[1] += 1
        elif value == 3:
            allCounts[2] += 1
        else:
            allCounts[3] += 1

    for index in range(len(allCounts)):
        if allCounts[index] != 0:
            compute += allCounts[index] / totalCount * math.log(allCounts[index] / totalCount, 2)

    # compute the entropy of all 4 classes combined
    entropySplit = -1 * compute
    return entropySplit


def findEntropyParent(totalCount, allCounts):
    """
    Find the entropy of the parent
    :param totalCount: the total points in the list
    :param allCounts: the list of class counts for each sample
    :return: the computed entropy
    """
    compute = 0
    for index in range(len(allCounts)):
        if allCounts[index] != 0:
            compute += allCounts[index] / totalCount * math.log(allCounts[index] / totalCount, 2)

    entropyUnsplit = -1 * compute
    return entropyUnsplit


def findInfoGain(pointList, totalCount, entropyUnsplit):
    """
    Function to compute information gain
    :param pointList: the list of samples
    :param totalCount: the total count of points in the list
    :param entropyUnsplit: the entropy of the parent
    :return: the list of characteristics of given sample
    """
    firstList = sorted(pointList, key=lambda elem: elem.x)        # sort the list of first attribute instances
    secondList = sorted(pointList, key=lambda elem: elem.y)       # sort the list of second attribute instances

    value1 = computations(firstList, totalCount, entropyUnsplit, attrib=1)
    value2 = computations(secondList, totalCount, entropyUnsplit, attrib=2)

    # check which attribute has a higher information gain and return that attribute features
    if value1[0] > value2[0]:
        for _ in pointList:
            if _.id == value1[2]:
                _.attribNumber = 1                                  # set the attribute number of the sample to 1
                return value1
    else:
        for _ in pointList:
            if _.id == value2[2]:
                _.attribNumber = 2                                  # set the attribute number of the sample to 2
                return value2


def buildTree(pointList, root=None, depth=0, check=0):
    """
    Function to build the tree
    :param pointList: the list of elements to be stored in the tree
    :param root: the root node
    :param depth: the depth of the tree
    :param check: a flag to check whether the recursive call came from the left subtree or right subtree
    :return: None
    """
    allCounts = [0, 0, 0, 0]
    totalCount = len(pointList)
    # iterate over the entire list of points and check the class count for each sample
    for _ in range(len(pointList)):
        if pointList[_].z == 1:
            allCounts[0] += 1
        elif pointList[_].z == 2:
            allCounts[1] += 1
        elif pointList[_].z == 3:
            allCounts[2] += 1
        else:
            allCounts[3] += 1

    # check if all the values in the list belong to the same class
    if len(pointList) == allCounts[0] or len(pointList) == allCounts[1] or\
            len(pointList) == allCounts[2] or len(pointList) == allCounts[3]:
        # create a new leaf node with the root features except the leaf value
        leaf = DataPoint(root.id, root.x, root.y, root.z)
        leaf.isLeafNode = True
        leaf.countClass = root.countClass
        leaf.attribNumber = 0
        leaf.threshold = root.threshold
        # check whether to set the new leaf node as the right child or the left child
        if check == 0:
            root.rightNode = leaf
        else:
            root.leftNode = leaf

        # check which class all points in the list belong to
        if len(pointList) == allCounts[0]:
            leaf.leafValue = 1
            return allCounts[0], leaf

        elif len(pointList) == allCounts[1]:
            leaf.leafValue = 2
            return allCounts[1], leaf

        elif len(pointList) == allCounts[2]:
            leaf.leafValue = 3
            return allCounts[2], leaf

        elif len(pointList) == allCounts[3]:
            leaf.leafValue = 4
            return allCounts[3], leaf

    else:
        entropyUnsplit = findEntropyParent(totalCount, allCounts)
        value = findInfoGain(pointList, totalCount, entropyUnsplit)
        elementId = value[2]
        for eachPoint in pointList:
            if eachPoint.id == elementId:
                root = eachPoint
        root.threshold = value[1]
        root.rightList = value[3]
        root.leftList = value[4]

        _, root.rightNode = buildTree(root.rightList, root, depth + 1, check=0)
        _, root.leftNode = buildTree(root.leftList, root, depth + 1, check=1)
        return allCounts, root


def computations(listSorted, totalCount, entropyUnsplit, attrib):
    """
    Function to perform all computations
    :param listSorted: the sorted list of attribute values
    :param totalCount: the total count of points in the list
    :param entropyUnsplit: the entropy of the parent
    :param attrib: a flag to check whether the first attribute has a better IG value than the second attribute
    :return: the list of sample characteristics
    """
    midValue = 0
    midAndIG = []
    # iterate over the entire list of sorted elements
    for index in range(len(listSorted)):
        listSorted[index].rightList, listSorted[index].leftList = [], []
        if index != len(listSorted) - 1:

            if attrib == 1:
                midValue = (listSorted[index].x + listSorted[index + 1].x) / 2   # calculate the middle value of adjacent instances
                # iterate over the entire list and check which instances are above and below the middle value
                # if they are above the middle value, append them to the right child list, else append them to the left child list
                for innerIndex in range(len(listSorted)):
                    if listSorted[innerIndex].x >= midValue:
                        listSorted[index].rightList.append(listSorted[innerIndex])
                    else:
                        listSorted[index].leftList.append(listSorted[innerIndex])
            else:
                midValue = (listSorted[index].y + listSorted[index + 1].y) / 2   # calculate the middle value of adjacent instances
                # iterate over the entire list and check which instances are above and below the middle value
                # if they are above the middle value, append them to the right child list, else append them to the left child list
                for innerIndex in range(len(listSorted)):
                    if listSorted[innerIndex].y >= midValue:
                        listSorted[index].rightList.append(listSorted[innerIndex])
                    else:
                        listSorted[index].leftList.append(listSorted[innerIndex])
        listSorted[index].countClass = countClass(listSorted)

        # store the total count of points or objects in the right node list and in the left node list
        totalCountRight = len(listSorted[index].rightList)
        totalCountLeft = len(listSorted[index].leftList)
        if totalCountRight == 0 or totalCountLeft == 0:
            continue

        entropyRight = findEntropyChild(totalCountRight, listSorted[index].rightList)         # compute the entropy of right child
        entropyLeft = findEntropyChild(totalCountLeft, listSorted[index].leftList)            # compute the entropy of left child

        # calculate the information gain as the difference in entropy of parent and entropy of both children
        IG = entropyUnsplit - ((totalCountRight / totalCount) * entropyRight + (totalCountLeft / totalCount) * entropyLeft)

        # append to a list the IG value and the middle value
        midAndIG.append([IG, midValue, listSorted[index].id, listSorted[index].rightList, listSorted[index].leftList])

    midAndIG.sort(key=lambda x: x[0])                                         # sort the list based on their IG values
    index, value = max(enumerate(midAndIG), key=operator.itemgetter(0))       # store the index and value of max. IG value

    return value


def classifySamples(sample, root):
    """
    Classify given sample for the decision plot
    :param sample: the sample to classify
    :param root: the root of the tree
    :return: the leaf value
    """
    if root.attribNumber == 1:
        if sample[0] > root.threshold:
            return classifySamples(sample, root.rightNode)
        else:
            return classifySamples(sample, root.leftNode)
    elif root.attribNumber == 2:
        if sample[1] > root.threshold:
            return classifySamples(sample, root.rightNode)
        else:
            return classifySamples(sample, root.leftNode)
    else:
        value = root.leafValue
        return value


def classifyFull(sampleList, pointList, root):
    """
    Classify the list of points and plot them on the decision plot
    :param sampleList: the list of points to draw the region
    :param pointList: the list of training data points
    :param root: the root of the decision tree
    :return:
    """
    correct1x, correct1y, correct2x, correct2y = [], [], [], []
    correct3x, correct3y, correct4x, correct4y = [], [], [], []
    for index in range(len(sampleList)):
        prediction = classifySamples(sampleList[index], root)
        if prediction == 1:
            correct1x.append(sampleList[index][0])
            correct1y.append(sampleList[index][1])

        if prediction == 2:
            correct2x.append(sampleList[index][0])
            correct2y.append(sampleList[index][1])

        if prediction == 3:
            correct3x.append(sampleList[index][0])
            correct3y.append(sampleList[index][1])

        if prediction == 4:
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
    plt.show()


def chiSquaredTest(object, significance):
    """
    Perform the chi-squared test on the given object to see whether splitting helps
    :param object: the object to be checked for significance
    :param significance: the significance value
    :return: True if the chi-squared value is lesser than the computed k value, False otherwise
    """
    inLeft = sum(object.leftNode.countClass) / sum(object.countClass)        # the total count of each class in the left node
    inRight = sum(object.rightNode.countClass) / sum(object.countClass)      # the total count of each class in the right node
    expectedRight, expectedLeft = [], []

    # iterate over the list of count values and compute the expected value in left and right nodes
    for _ in range(len(object.countClass)):
        expectedLeft.append(object.countClass[_] * inLeft)
        expectedRight.append(object.countClass[_] * inRight)

    k = 0
    # iterate over the list of count values to compute value of k
    for index in range(len(object.countClass)):
        if expectedLeft[index] != 0:
            k += math.pow(expectedLeft[index] - object.leftNode.countClass[index], 2) / expectedLeft[index]
        if expectedRight[index] != 0:
            k += math.pow(expectedRight[index] - object.rightNode.countClass[index], 2) / expectedRight[index]
    # check for significance
    print(k)
    if significance == 0.01 and k < 11.345:
        return k, True
    if significance == 0.05 and k < 7.815:
        return k, True
    else:
        return k, False


def pruning(root, depth, nodeDepth, significance):
    """
    Function to prune the unwanted branches of the decision tree
    :param root: the root node of the tree
    :param depth: the depth of the tree
    :param nodeDepth: the depth at the node
    :param significance: the significance level at which to test for pruning
    :return: None
    """
    # check whether the root is a leaf node or not

    if root is not None:
        if root.rightNode.isLeafNode and root.leftNode.isLeafNode:
                valueOfK, bool = chiSquaredTest(root, significance)
                print(valueOfK)
                if bool:
                    root.leafValue = max(root.rightNode.countClass, root.leftNode.countClass)
                    root.rightNode = None
                    root.leftNode = None
                    root.isLeafNode = True
                return
    else:
        if root is not None:
            if not root.leftNode.isLeafNode:
                pruning(root.leftNode, depth + 1, nodeDepth, significance)
            if not root.rightNode.isLeafNode:
                pruning(root.rightNode, depth + 1, nodeDepth, significance)


def writeToFile(pointer, root, depth=0):
    """
    Function to write the decision tree to a new file
    :param pointer: the pointer to the file
    :param root: the root node of the decision tree
    :param depth: the current depth of the tree
    :return: None
    """
    # check if the root is None
    if root is None:
        line = str('\t' * depth) + 'end' + '\n'
        pointer.write(line)
        return
    # check if the root is a leaf node or not
    if not root.isLeafNode:
        # write the attribute number, threshold and leaf value if present
        line = str('\t' * depth) + str(root.attribNumber) + ',' + str(root.threshold) + ',' + str(0) + '\n'
    else:
        # compute the leaf value
        value = [x for x in range(len(root.countClass)) if root.countClass[x] != 0]
        line = str('\t' * depth) + str(root.attribNumber) + ',' + str(root.threshold) + ',' + str(root.leafValue) + '\n'
        pointer.write(line)
        return

    pointer.write(line)
    # recursively call the left and right subtree
    writeToFile(pointer, root.leftNode, depth + 1)
    writeToFile(pointer, root.rightNode, depth + 1)


def findMaxDepth(root):
    """
    Find the max depth of the decision tree
    :param root: the root of the tree
    :return: the max depth
    """
    if root.leftNode is None and root.rightNode is None:
        return 0
    return 1 + max(findMinDepth(root.leftNode), findMinDepth(root.rightNode))


def findMinDepth(root):
    """
    Find the min depth of the tree
    :param root: the root of the decision tree
    :return: the min depth
    """
    if root.leftNode is None and root.rightNode is None:
        return 0
    return 1 + min(findMinDepth(root.leftNode), findMinDepth(root.rightNode))


def createListForPlotting():
    """
    Create the list of x and y values for plotting the graph
    :return: the list of x and y points
    """
    stepSize = 0.001
    data = []
    for index in range(1000):
        index /= 1000
        for idx in range(1000):
            idx /= 1000
            data.append([index + stepSize, idx + stepSize])
    return data


def printTree(root, depth=0):
    """
    Prints the decision tree given the root node
    :param root: the root node of the tree
    :param depth: the depth of the tree
    :return: None
    """
    if root is not None:
        if root.leafValue == float(0):
            print('\t' * depth + 'Attribute number: ' + str(root.attribNumber) + ' Threshold value: ' + str(root.threshold) + ' Non-terminal node')
        else:
            print('\t' * depth + 'Attribute number: ' + str(root.attribNumber) + ' Threshold value: ' + str(root.threshold) + ' Class value: ' + str(root.leafValue))
        printTree(root.leftNode, depth+1)
        printTree(root.rightNode, depth+1)


def main():
    """
    The main function
    :return: None
    """
    if len(sys.argv) < 2:
        print("input training file not supplied!")
        exit(0)
    filename = sys.argv[1]
    pointList = readInput(filename)

    print('-------------- Before pruning ------------------')
    print()
    count, root = buildTree(pointList)
    printTree(root)
    print()

    print('******** Depth of the tree *********')
    print()
    maxDepth = findMaxDepth(root)
    minDepth = findMinDepth(root)
    print('Min depth: ', minDepth)
    print('Max depth: ', maxDepth)
    print()

    data = createListForPlotting()
    classifyFull(data, pointList, root)

    pointer = open('DT_before.txt', 'w')
    writeToFile(pointer, root)

    print('---------------- After pruning -----------------')
    print()
    pruning(root, 0, maxDepth - 1, significance=0.01)

    printTree(root)
    print()
    pointer = open('DT_after.txt', 'w')
    writeToFile(pointer, root)

    print('******** Depth of the tree *********')
    print()
    maxDepth = findMaxDepth(root)
    minDepth = findMinDepth(root)
    print('Min depth: ', minDepth)
    print('Max depth: ', maxDepth)
    print()

    prunedData = createListForPlotting()
    classifyFull(prunedData, pointList, root)

    pointer.close()


if __name__ == '__main__':
    main()
