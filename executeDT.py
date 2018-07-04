from beautifultable import BeautifulTable
import matplotlib.pyplot as plt
import sys
from trainDT import readInput, DataPoint, print_tree, create_list_for_plotting


def set_left_and_right(object_list):
    """
    Set the left and right children for each object in the given list
    :param object_list: the list of objects whose children need to be set
    :return: the updated list of objects
    """
    for index in range(len(object_list)):
        if object_list[index].id == 0:
            for index2 in range(len(object_list)):
                if object_list[index2].id == 1:
                    object_list[index].leftnode = object_list[index2]
            for index3 in range(len(object_list)):
                if object_list[index3].id == 12:
                    object_list[index].rightnode = object_list[index3]

        elif object_list[index].id == 1:
            for index2 in range(len(object_list)):
                if object_list[index2].id == 2:
                    object_list[index].leftnode = object_list[index2]
            for index3 in range(len(object_list)):
                if object_list[index3].id == 7:
                    object_list[index].rightnode = object_list[index3]

        elif object_list[index].id == 2:
            for index2 in range(len(object_list)):
                if object_list[index2].id == 3:
                    object_list[index].leftnode = object_list[index2]
                    object_list[index2].is_leafnode = True
            for index3 in range(len(object_list)):
                if object_list[index3].id == 4:
                    object_list[index].rightnode = object_list[index3]

        elif object_list[index].id == 4:
            for index2 in range(len(object_list)):
                if object_list[index2].id == 5:
                    object_list[index].leftnode = object_list[index2]
                    object_list[index2].is_leafnode = True
            for index3 in range(len(object_list)):
                if object_list[index3].id == 6:
                    object_list[index].rightnode = object_list[index3]
                    object_list[index3].is_leafnode = True

        elif object_list[index].id == 7:
            for index2 in range(len(object_list)):
                if object_list[index2].id == 8:
                    object_list[index].leftnode = object_list[index2]
                    object_list[index2].is_leafnode = True
            for index3 in range(len(object_list)):
                if object_list[index3].id == 9:
                    object_list[index].rightnode = object_list[index3]

        elif object_list[index].id == 9:
            for index2 in range(len(object_list)):
                if object_list[index2].id == 10:
                    object_list[index].leftnode = object_list[index2]
                    object_list[index2].is_leafnode = True
            for index3 in range(len(object_list)):
                if object_list[index3].id == 11:
                    object_list[index].rightnode = object_list[index3]
                    object_list[index3].is_leafnode = True

        elif object_list[index].id == 12:
            for index2 in range(len(object_list)):
                if object_list[index2].id == 13:
                    object_list[index].leftnode = object_list[index2]
            for index3 in range(len(object_list)):
                if object_list[index3].id == 18:
                    object_list[index].rightnode = object_list[index3]
                    object_list[index3].is_leafnode = True

        elif object_list[index].id == 13:
            for index2 in range(len(object_list)):
                if object_list[index2].id == 14:
                    object_list[index].leftnode = object_list[index2]
                    object_list[index2].is_leafnode = True
            for index3 in range(len(object_list)):
                if object_list[index3].id == 15:
                    object_list[index].rightnode = object_list[index3]

        elif object_list[index].id == 15:
            for index2 in range(len(object_list)):
                if object_list[index2].id == 16:
                    object_list[index].leftnode = object_list[index2]
                    object_list[index2].is_leafnode = True
            for index3 in range(len(object_list)):
                if object_list[index3].id == 17:
                    object_list[index].rightnode = object_list[index3]
                    object_list[index3].is_leafnode = True
        else:
            object_list[index].rightnode = object_list[index].leftnode = None

    return object_list


def create_tree_list(dt_file):
    """
    Creates a list of tree objects by reading in the decision tree file created by the trainDT.py
    :param dt_file: the file created by trainDT.py, which contains the decision tree object
    :return: the list of tree objects
    """
    f = open(dt_file, 'r')
    data = f.readlines()
    tree_list = []
    for line in data:
        tree_list.append(line.strip().split(','))
    return tree_list


def create_objects(tree_list):
    """
    For a given list, create an object and set all its features
    :param tree_list: the list containing the tree elements in the form of a list
    :return: the list of objects for the tree
    """
    datapoint_list = []
    for index in range(len(tree_list)):
        if tree_list[index][2] == '0':
            datapoint = DataPoint(index, 0, 0, 0)
            datapoint.attribnumber = float(tree_list[index][0])
            datapoint.threshold = float(tree_list[index][1])
            datapoint.leafvalue = int(tree_list[index][2])
            datapoint_list.append(datapoint)
        else:
            datapoint = DataPoint(index, 0, 0, 0)
            datapoint.leafvalue = int(tree_list[index][2])
            datapoint.is_leafnode = True
            datapoint_list.append(datapoint)
    return datapoint_list


def classify_samples(sample, root):
    """
    Given the sample, classify it into one of the 4 classes
    :param sample: the sample to be classified
    :param root: the root of the decision tree
    :return: the leaf value containing the class
    """
    # check which attribute to split on and based on that recursively call
    if root.attribnumber == 1:
        if sample.x > root.threshold:
            return classify_samples(sample, root.rightnode)
        else:
            return classify_samples(sample, root.leftnode)
    elif root.attribnumber == 2:
        if sample.y > root.threshold:
            return classify_samples(sample, root.rightnode)
        else:
            return classify_samples(sample, root.leftnode)
    else:
        value = root.leafvalue
        return value


def classify_for_plot(sample, root):
    """
    Classify given sample for the decision plot
    :param sample: the sample to classify
    :param root: the root of the tree
    :return: the leaf value
    """
    if root.attribnumber == 1:
        if sample[0] > root.threshold:
            return classify_for_plot(sample, root.rightnode)
        else:
            return classify_for_plot(sample, root.leftnode)
    elif root.attribnumber == 2:
        if sample[1] > root.threshold:
            return classify_for_plot(sample, root.rightnode)
        else:
            return classify_for_plot(sample, root.leftnode)
    else:
        value = root.leafvalue
        return value


def classify_full(sample_list, point_list, root):
    """
    Classify the list of points and plot them on the decision plot
    :param sample_list: the list of points to draw the region
    :param point_list: the list of testing data points
    :param root: the root of the decision tree
    :return:
    """
    correct1x, correct1y, correct2x, correct2y, correct3x, correct3y, correct4x, correct4y = [], [], [], [], [], [], [], []
    for index in range(len(sample_list)):
        prediction = classify_for_plot(sample_list[index], root)
        if prediction == 1:
            correct1x.append(sample_list[index][0])
            correct1y.append(sample_list[index][1])

        if prediction == 2:
            correct2x.append(sample_list[index][0])
            correct2y.append(sample_list[index][1])

        if prediction == 3:
            correct3x.append(sample_list[index][0])
            correct3y.append(sample_list[index][1])

        if prediction == 4:
            correct4x.append(sample_list[index][0])
            correct4y.append(sample_list[index][1])

    plt.plot(correct1x, correct1y, 'green')
    plt.plot(correct2x, correct2y, 'blue')
    plt.plot(correct3x, correct3y, 'red')
    plt.plot(correct4x, correct4y, 'black')
    point1x, point1y, point2x, point2y, point3x, point3y, point4x, point4y = [], [], [], [], [], [], [], []
    for index in range(len(point_list)):
       if point_list[index].z == 1:
           point1x.append(point_list[index].x)
           point1y.append(point_list[index].y)
       elif point_list[index].z == 2:
           point2x.append(point_list[index].x)
           point2y.append(point_list[index].y)
       elif point_list[index].z == 3:
           point3x.append(point_list[index].x)
           point3y.append(point_list[index].y)
       else:
           point4x.append(point_list[index].x)
           point4y.append(point_list[index].y)
    plt.plot(point1x, point1y, 'c*', label='Bolt')
    plt.plot(point2x, point2y, 'y^', label='Nut')
    plt.plot(point3x, point3y, 'mx', label='Ring')
    plt.plot(point4x, point4y, 'w+', label='Scrap')
    plt.title("Decision boundary")
    plt.xlabel("Six fold rotational symmetry")
    plt.ylabel("Eccentricity")
    plt.legend(loc='upper right')
    plt.show()


def classify_all_samples(sample_list, root):
    """
    Function to classify the given list of samples using the classifier created above
    :param sample_list: the list of samples
    :param root: the root node of the tree
    :return: the accuracy of the classifier and its mean per class accuracy
    """
    correct, incorrect = 0, 0
    mean_per_class = 0
    correct_class, incorrect_class = [0 for idx in range(4)], [0 for idx in range(4)]   # create a 2D matrix of 4x4 with all 0 elements
    # iterate over the entire list of samples
    for index in range(len(sample_list)):
        class_value = classify_samples(sample_list[index], root)                        # get the predicted class of a given sample
        if class_value == sample_list[index].z:                                         # check if the predicted class is the same as actual class
            correct += 1
            # set the count of correctly classified instance in the list at its corresponding location
            for idx in range(4):
                if sample_list[index].z == idx + 1:
                    correct_class[idx] += 1
        else:
            incorrect += 1
            # set the count of incorrectly classified instance in the list at its corresponding location
            for idx in range(4):
                if sample_list[index].z == idx + 1:
                    incorrect_class[idx] += 1

    accuracy = correct / (correct + incorrect) * 100                                    # check the accuracy of the classifier
    # compute the mean per class accuracy
    for idx in range(4):
        mean_per_class += correct_class[idx] / (correct_class[idx] + incorrect_class[idx])
    mean_per_class /= 4

    return mean_per_class, accuracy


def build_confusion_matrix(sample_list, root):
    """
    Function to build the confusion matrix
    :param sample_list: the list of all samples
    :param root: the root node of the tree
    :return: the confusion matrix
    """
    confusion_matrix = [[0 for idx in range(4)] for idx in range(4)]                # create a 2D matrix of 4x4 size with all 0 elements
    # iterate over the entire list of samples to check whether the class is correctly predicted or not
    for index in range(len(sample_list)):
        class_value = classify_samples(sample_list[index], root)
        actual_class_value = sample_list[index].z
        confusion_matrix[int(class_value) - 1][int(actual_class_value) - 1] += 1

    return confusion_matrix


def print_table(confusion_matrix):
    """
    Helper function that is used to print the confusion matrix
    :param confusion_matrix: list of lists containing the confusion matrix
    :return: None
    """
    table = BeautifulTable()
    classes = ["Bolt", "Nut", "Ring", "Scrap"]
    table.column_headers = classes
    for line in confusion_matrix:
        table.append_row(line)
    table.insert_column(0, "Predicted (down) \ Actual (across)", classes)
    print(table)


def calculate_profit(confusion_matrix):
    """
    Compute the total profit obtained by using the classifier
    :param confusion_matrix: the confusion matrix to compute profit
    :return: the total profit
    """
    profit_matrix = [[20, -7, -7, -7], [-7, 15, -7, -7], [-7, -7, 5, -7], [-3, -3, -3, -3]]                     # the given profit matrix
    total_profit = 0
    for outerindex in range(len(profit_matrix)):
        for innerindex in range(len(profit_matrix)):
            total_profit += profit_matrix[outerindex][innerindex] * confusion_matrix[outerindex][innerindex]    # compute profit obtained for each class

    return total_profit


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
    data_file = sys.argv[2]
    point_list = readInput(data_file)

    print('------------- Before pruning --------------')

    tree_list = create_tree_list(filename)
    object_list = create_objects(tree_list)
    new_object_list = set_left_and_right(object_list)
    root_object = new_object_list[0]
    confusion_matrix = build_confusion_matrix(point_list, root_object)
    print_table(confusion_matrix)
    data = create_list_for_plotting()
    classify_full(data, point_list, root_object)
    print()
    print('********** Recognition rate ***********')
    mpc_accuracy, accuracy = classify_all_samples(point_list, root_object)
    print('Mean per class accuracy is: ', str(mpc_accuracy))
    print('Accuracy is: ', str(accuracy))
    print('***************************************')
    print()
    print('************ Profit ***********')
    print('Total profit computed is:', str(calculate_profit(confusion_matrix)))
    print('*******************************')

    print('----------- After pruning ------------')

    tree_list_pruned = create_tree_list(filename)
    object_list_pruned = create_objects(tree_list_pruned)
    new_object_list_pruned = set_left_and_right(object_list_pruned)
    root_object_pruned = new_object_list_pruned[0]
    confusion_matrix = build_confusion_matrix(point_list, root_object_pruned)

    print_table(confusion_matrix)
    data_pruned = create_list_for_plotting()
    classify_full(data_pruned, point_list, root_object)
    print()
    print('********** Recognition rate ***********')
    mpc_accuracy, accuracy = classify_all_samples(point_list, root_object)
    print('Mean per class accuracy is: ', str(mpc_accuracy))
    print('Accuracy is: ', str(accuracy))
    print('***************************************')
    print()
    print('************ Profit ***********')
    print('Total profit computed is:', str(calculate_profit(confusion_matrix)))
    print('*******************************')


if __name__ == '__main__':
    main()