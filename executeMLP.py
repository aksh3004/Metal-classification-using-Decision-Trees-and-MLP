"""
Authors: Akshay Karki and Krishna Tippur Gururaj
"""
from trainMLP import Node, Data, read_file, compute_sigmoid
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from beautifultable import BeautifulTable


def read_from_file(filename):
    """
    function used to read data from supplied file
    :param filename: filename
    :return: dictionary of contents of file
    """
    contents = {}
    with open(filename) as f:
        for line in f:
            li = line.strip().split(',')
            contents[int(li[0])] = [float(i) for i in li[1:]]
    return contents


def recreate_model(filename, hidden_nodes_count):
    """
    function which recreates the MLP by reading network weights from the given CSV file
    :param filename: file containing weight information of the network
    :param hidden_nodes_count: number of nodes in the hidden layer
    :return: configuration of the network [nodes' information]
    """
    contents = read_from_file(filename)

    #  initialize two input nodes
    x1 = Node(1, hidden_nodes_count)
    x2 = Node(x1.iden + 1, hidden_nodes_count)

    #  initialize list of hidden layer's nodes
    hidden_nodes = [Node(x + 3, 4) for x in range(hidden_nodes_count)]

    #  initialize four output nodes
    y1 = Node(hidden_nodes_count + 3, 0)
    y2 = Node(y1.iden + 1, 0)
    y3 = Node(y2.iden + 1, 0)
    y4 = Node(y3.iden + 1, 0)
    y_nodes = [y1, y2, y3, y4]

    #  initialize two bias nodes
    b1 = Node(y_nodes[3].iden + 1, hidden_nodes_count)
    b2 = Node(b1.iden + 1, 4)

    #  set weights from file
    x1.connections = contents[x1.iden]
    x2.connections = contents[x2.iden]
    b1.connections = contents[b1.iden]
    b2.connections = contents[b2.iden]

    for idx in range(len(hidden_nodes)):
        hn = hidden_nodes[idx]
        hn.connections = contents[hn.iden]

    return x1, x2, b1, b2, hidden_nodes


def execute_mlp(weight_file, data):
    """
    function that executes MLP using given weights on the supplied data
    :param weight_file: CSV file containing the network weights
    :param data: list of test data values
    :return: confusion matrix, accuracy of model, mean per class accuracy value of model
    """
    confusion_matrix = [[0, 0, 0, 0] for _ in range(4)]
    x1, x2, b1, b2, hidden_nodes = recreate_model(weight_file, 5)

    #  initialize four output nodes
    y1 = Node(len(hidden_nodes) + 3, 0)
    y2 = Node(y1.iden + 1, 0)
    y3 = Node(y2.iden + 1, 0)
    y4 = Node(y3.iden + 1, 0)
    y_nodes = [y1, y2, y3, y4]
    actual_data_classes = [0, 0, 0, 0]

    for data_point in data:

        actual_data_classes[int(data_point.class_val) - 1] += 1

        #  forward propagation
        for val in range(len(hidden_nodes)):
            hn = hidden_nodes[val]
            hn.value = compute_sigmoid(b1.connections[val] + (x1.connections[val] * data_point.rot_sym) +
                                       (x2.connections[val] * data_point.ecc))

        #  add bias to each output
        y_nodes[0].value += b2.connections[0]
        y_nodes[1].value += b2.connections[1]
        y_nodes[2].value += b2.connections[2]
        y_nodes[3].value += b2.connections[3]

        #  compute output value
        for val in range(len(hidden_nodes)):
            y_nodes[0].value += (hidden_nodes[val].connections[0] * hidden_nodes[val].value)
            y_nodes[1].value += (hidden_nodes[val].connections[1] * hidden_nodes[val].value)
            y_nodes[2].value += (hidden_nodes[val].connections[2] * hidden_nodes[val].value)
            y_nodes[3].value += (hidden_nodes[val].connections[3] * hidden_nodes[val].value)

        #  convert output using sigmoid
        y_nodes[0].value = compute_sigmoid(y_nodes[0].value)
        y_nodes[1].value = compute_sigmoid(y_nodes[1].value)
        y_nodes[2].value = compute_sigmoid(y_nodes[2].value)
        y_nodes[3].value = compute_sigmoid(y_nodes[3].value)

        y_values = [y_nodes[0].value, y_nodes[1].value, y_nodes[2].value, y_nodes[3].value]

        if np.argmax(y_values) == int(data_point.class_val) - 1:
            confusion_matrix[int(data_point.class_val) - 1][int(data_point.class_val) - 1] += 1
        else:
            confusion_matrix[int(np.argmax(y_values))][int(data_point.class_val) - 1] += 1
    acc = (confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2]
           + confusion_matrix[3][3]) / len(data)
    mpc_acc = ((confusion_matrix[0][0] / actual_data_classes[0]) +
               (confusion_matrix[1][1] / actual_data_classes[1]) +
               (confusion_matrix[2][2] / actual_data_classes[2]) +
               (confusion_matrix[3][3] / actual_data_classes[3])) / 4
    return confusion_matrix, acc, mpc_acc


def compute_profit(confusion_matrix):
    """
    function used to compute profit that the given model obtains by classifying the input data
    :param confusion_matrix: confusion matrix showing the performance of the classification
    :return: profit of the model
    """
    profit_matrix = [[20, -7, -7, -7], [-7, 15, -7, -7], [-7, -7, 5, -7], [-3, -3, -3, -3]]
    profit = 0
    for row_num in range(len(confusion_matrix)):
        for col_num in range(len(profit_matrix[0])):
            profit += confusion_matrix[row_num][col_num] * profit_matrix[row_num][col_num]
    return profit


def print_table(confusion_matrix):
    """
    helper function that is used to print the confusion matrix
    :param confusion_matrix: list of lists containing the confusion matrix
    :return: None
    """
    table = BeautifulTable()
    classes = ["Bolt", "Nut", "Ring", "Scrap"]
    table.column_headers = classes
    for line in confusion_matrix:
        table.append_row(line)
    table.insert_column(0, "Assigned (down) \ Actual (across)", classes)
    print(table)
    return


def draw_classification_regions(weight_file, test_data):
    """
    this function is used to plot the classification regions based on a grid of equally spaced points
    in the region (0, 0) to (1, 1), using a different color to represent the assigned class. The test data
    is then plotted on top of the regions to showcase the accuracy of the model
    :param weight_file: network configuration of the model
    :param test_data: test data
    :return: None
    """
    x1, x2, b1, b2, hidden_nodes = recreate_model(weight_file, 5)

    #  initialize four output nodes
    y1 = Node(len(hidden_nodes) + 3, 0)
    y2 = Node(y1.iden + 1, 0)
    y3 = Node(y2.iden + 1, 0)
    y4 = Node(y3.iden + 1, 0)
    y_nodes = [y1, y2, y3, y4]

    data = []
    data_split_by_class = [[[], []], [[], []], [[], []], [[], []]]

    for x in range(100):
        for y in range(100):
            data.append([float(x * 0.01), float(y * 0.01)])

    for data_point in data:

        #  forward propagation
        for val in range(len(hidden_nodes)):
            hn = hidden_nodes[val]
            hn.value = compute_sigmoid(b1.connections[val] + (x1.connections[val] * data_point[0]) +
                                       (x2.connections[val] * data_point[1]))

        #  add bias to each output
        y_nodes[0].value += b2.connections[0]
        y_nodes[1].value += b2.connections[1]
        y_nodes[2].value += b2.connections[2]
        y_nodes[3].value += b2.connections[3]

        #  compute output value
        for val in range(len(hidden_nodes)):
            y_nodes[0].value += (hidden_nodes[val].connections[0] * hidden_nodes[val].value)
            y_nodes[1].value += (hidden_nodes[val].connections[1] * hidden_nodes[val].value)
            y_nodes[2].value += (hidden_nodes[val].connections[2] * hidden_nodes[val].value)
            y_nodes[3].value += (hidden_nodes[val].connections[3] * hidden_nodes[val].value)

        #  convert output using sigmoid
        y_nodes[0].value = compute_sigmoid(y_nodes[0].value)
        y_nodes[1].value = compute_sigmoid(y_nodes[1].value)
        y_nodes[2].value = compute_sigmoid(y_nodes[2].value)
        y_nodes[3].value = compute_sigmoid(y_nodes[3].value)

        y_values = [y_nodes[0].value, y_nodes[1].value, y_nodes[2].value, y_nodes[3].value]
        pred_class = int(np.argmax(y_values) - 1)
        data_split_by_class[pred_class][0].append(data_point[0])
        data_split_by_class[pred_class][1].append(data_point[1])

    test_split_by_class = [[[], []], [[], []], [[], []], [[], []]]
    for point in test_data:
        test_split_by_class[int(point.class_val) - 1][0].append(point.rot_sym)
        test_split_by_class[int(point.class_val) - 1][1].append(point.ecc)

    plt.figure(1)
    plt.scatter(data_split_by_class[0][0], data_split_by_class[0][1], color='lightgreen')
    plt.scatter(data_split_by_class[1][0], data_split_by_class[1][1], color='lightblue')
    plt.scatter(data_split_by_class[2][0], data_split_by_class[2][1], color='orange')
    plt.scatter(data_split_by_class[3][0], data_split_by_class[3][1], color='lightyellow')

    plt.scatter(test_split_by_class[0][0], test_split_by_class[0][1], color='magenta', marker='o', label='Bolt')
    plt.scatter(test_split_by_class[1][0], test_split_by_class[1][1], color='yellow', marker='v', label='Nut')
    plt.scatter(test_split_by_class[2][0], test_split_by_class[2][1], color='green', marker='^', label='Ring')
    plt.scatter(test_split_by_class[3][0], test_split_by_class[3][1], color='black', marker='<', label='Scrap')
    plt.title("Decision boundary")
    plt.xlabel("Six fold rotational symmetry")
    plt.ylabel("Eccentricity")
    plt.legend(loc='upper right')
    plt.show()
    return


def main():
    if len(sys.argv) < 3:
        print("incorrect cmd line params passed")
        print("correct usage: python3 executeMLP.py <weights file> <test data file>")
        print("eg: python3 executeMLP.py MLPweights10000.csv test_data.csv")
        exit(1)
    if not os.path.isfile(sys.argv[1]):
        print("error! File", sys.argv[1], "not found")
        exit(1)
    if not os.path.isfile(sys.argv[2]):
        print("error! File", sys.argv[2], "not found")
        exit(1)
    data = read_file(sys.argv[2])
    confusion_matrix, acc, mpc_acc = execute_mlp(sys.argv[1], data)
    print("----- Confusion Matrix -----")
    print_table(confusion_matrix)
    profit = compute_profit(confusion_matrix)
    print("Profit:", profit)
    print("Accuracy:", acc)
    print("Mean per class accuracy:", mpc_acc)
    draw_classification_regions(sys.argv[1], data)
    return


if __name__ == '__main__':
    main()
