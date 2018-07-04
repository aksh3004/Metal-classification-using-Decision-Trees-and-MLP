"""
Authors: Akshay Karki and Krishna Tippur Gururaj
Date: 12/03/2017
"""
import os
import sys
import random
import math
import matplotlib.pyplot as plt


class Data:
    """
    this class holds the labelled data for this implementation
    """
    __slots__ = 'rot_sym', 'ecc', 'class_val'

    def __init__(self, val1, val2, val3):
        self.rot_sym = val1
        self.ecc = val2
        self.class_val = val3


class Node:
    """
    this class is used to represent a node in the multi-layer perceptron
    """
    __slots__ = 'iden', 'connections', 'value'

    def __init__(self, ident, no_of_connections):
        self.iden = ident
        self.connections = [random.uniform(-1, 1) for _ in range(no_of_connections)]
        self.value = 0


def read_file(filename):
    """
    helper function used to read labelled CSV data
    :param filename: CSV file
    :return: list of data points
    """
    data_contents = []
    with open(filename) as f:
        for line in f:
            if line is '\n':
                continue
            li = line.strip().split(',')
            data_contents.append(Data(float(li[0]), float(li[1]), float(li[2])))
    return data_contents


def write_to_file(filename, data):
    """
    helper function used to write the network weights to file
    :param filename: file in which the weights are to be written
    :param data: network weights
    :return: None
    """
    with open(filename, 'a+') as f:
        f.write(str(data.iden) + ",")
        f.write(','.join(str(i) for i in data.connections))
        f.write('\n')
    return


def compute_sigmoid(val):
    """
    which runs input through a sigmoid and returns output
    :param val: input
    :return: sigmoid value of input [lies between 0 and 1]
    """
    try:
        ret = 1 / (1 + math.exp(-val))
    except OverflowError:
        print("exception:", val)
        raise
    return ret


def compute_error_at_output(true_val, predicted_val_list):
    """
    function to compute error in the prediction of model
    :param true_val: actual class of data point
    :param predicted_val_list: predicted class of data point by model
    :return: error vector
    """
    a, b, c, d = 0, 0, 0, 0
    if true_val == 1:
        a = (predicted_val_list[0] - 1) * predicted_val_list[0] * (1 - predicted_val_list[0])
        b = (predicted_val_list[1] - 0) * predicted_val_list[1] * (1 - predicted_val_list[1])
        c = (predicted_val_list[2] - 0) * predicted_val_list[2] * (1 - predicted_val_list[2])
        d = (predicted_val_list[3] - 0) * predicted_val_list[3] * (1 - predicted_val_list[3])
    elif true_val == 2:
        a = (predicted_val_list[0] - 0) * predicted_val_list[0] * (1 - predicted_val_list[0])
        b = (predicted_val_list[1] - 1) * predicted_val_list[1] * (1 - predicted_val_list[1])
        c = (predicted_val_list[2] - 0) * predicted_val_list[2] * (1 - predicted_val_list[2])
        d = (predicted_val_list[3] - 0) * predicted_val_list[3] * (1 - predicted_val_list[3])
    elif true_val == 3:
        a = (predicted_val_list[0] - 0) * predicted_val_list[0] * (1 - predicted_val_list[0])
        b = (predicted_val_list[1] - 0) * predicted_val_list[1] * (1 - predicted_val_list[1])
        c = (predicted_val_list[2] - 1) * predicted_val_list[2] * (1 - predicted_val_list[2])
        d = (predicted_val_list[3] - 0) * predicted_val_list[3] * (1 - predicted_val_list[3])
    elif true_val == 4:
        a = (predicted_val_list[0] - 0) * predicted_val_list[0] * (1 - predicted_val_list[0])
        b = (predicted_val_list[1] - 0) * predicted_val_list[1] * (1 - predicted_val_list[1])
        c = (predicted_val_list[2] - 0) * predicted_val_list[2] * (1 - predicted_val_list[2])
        d = (predicted_val_list[3] - 1) * predicted_val_list[3] * (1 - predicted_val_list[3])
    else:
        print("unexpected value!!")
        exit(100)
    return [a, b, c, d]


def calculate_sse_for_sample(true_val, pred_val):
    """
    helper function used to compute sum of squared error for a data point
    :param true_val: actual class of data point
    :param pred_val: predicted class of data point
    :return: sse
    """
    if true_val == 1:
        return math.pow((pred_val[0] - 1), 2) + math.pow(pred_val[1], 2) + \
               math.pow(pred_val[2], 2) + math.pow(pred_val[3], 2)
    elif true_val == 2:
        return math.pow(pred_val[0], 2) + math.pow((pred_val[1] - 1), 2) + \
               math.pow(pred_val[2], 2) + math.pow(pred_val[3], 2)
    elif true_val == 3:
        return math.pow(pred_val[0], 2) + math.pow(pred_val[1], 2) + \
               math.pow((pred_val[2] - 1), 2) + math.pow(pred_val[3], 2)
    elif true_val == 4:
        return math.pow(pred_val[0], 2) + math.pow(pred_val[1], 2) + \
               math.pow(pred_val[2], 2) + math.pow((pred_val[3] - 1), 2)
    return 0


def train_mlp(data, hidden_nodes_count, epoch, output_file="MLPweights"):
    """
    function which implements the training of the multi-layer perceptron
    :param data: training dataset
    :param hidden_nodes_count: number of nodes in the hidden layer
    :param epoch: number of learning iterations that the model is to undergo
    :param output_file: file to which the final network weights are written
    :return: sse for the entire learning
    """
    required_epochs = [0, 10, 100, 1000, 10000]
    alpha = 0.01
    sse_list = []

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

    #  initialize weight vectors randomly
    for idx in range(epoch):
        sse = 0

        for data_point in data:

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

            #  compute errors at output layer
            output_layer_errors = compute_error_at_output(data_point.class_val, [y_nodes[0].value,
                                                                                 y_nodes[1].value, y_nodes[2].value,
                                                                                 y_nodes[3].value])

            sse += calculate_sse_for_sample(data_point.class_val, [y_nodes[0].value, y_nodes[1].value,
                                                                   y_nodes[2].value, y_nodes[3].value])
            #  back propagation starts here

            #  compute errors at hidden layer
            hidden_layer_errors = []
            for val in range(len(hidden_nodes)):
                d = ((output_layer_errors[0] * hidden_nodes[val].connections[0]) +
                     (output_layer_errors[1] * hidden_nodes[val].connections[1]) +
                     (output_layer_errors[2] * hidden_nodes[val].connections[2]) +
                     (output_layer_errors[3] * hidden_nodes[val].connections[3])) * \
                    hidden_nodes[val].value * (1 - hidden_nodes[val].value)
                hidden_layer_errors.append(d)

            #  update bias network weights
            for val in range(hidden_nodes_count):
                b1.connections[val] -= (alpha * hidden_layer_errors[val])

            for val in range(len(y_nodes)):
                b2.connections[val] -= (alpha * output_layer_errors[val])

            #  update hidden layer network weights
            for val in range(len(hidden_nodes)):
                hn = hidden_nodes[val]
                for conn_val in range(len(hn.connections)):
                    hn.connections[conn_val] -= (alpha * output_layer_errors[conn_val] * hn.value)

            #  update weights towards the input nodes
            for val in range(len(hidden_nodes)):
                x1.connections[val] -= (alpha * hidden_layer_errors[val] * data_point.rot_sym)
                x2.connections[val] -= (alpha * hidden_layer_errors[val] * data_point.ecc)
        sse_list.append(sse)
        filename = output_file + str(idx) + ".csv"
        if idx in required_epochs:
            if os.path.isfile(filename):
                print("old weights file", filename, "present. Deleting it ..")
                os.remove(filename)

            write_to_file(filename, x1)
            write_to_file(filename, x2)
            write_to_file(filename, b1)
            write_to_file(filename, b2)
            for hn in hidden_nodes:
                write_to_file(filename, hn)

    #  for epoch 10,000
    filename = output_file + str(epoch) + ".csv"
    if os.path.isfile(filename):
        print("old weights file", filename, "present. Deleting it ..")
        os.remove(filename)
    write_to_file(filename, x1)
    write_to_file(filename, x2)
    write_to_file(filename, b1)
    write_to_file(filename, b2)
    for hn in hidden_nodes:
        write_to_file(filename, hn)

    return sse_list


def main():
    if len(sys.argv) < 2:
        print("input training file not supplied!")
        exit(0)
    if not os.path.isfile(sys.argv[1]):
        print("input file " + sys.argv[1] + " not found!")
        exit(0)
    data = read_file(sys.argv[1])
    sse_list = train_mlp(data, 5, 10000)
    epoch_list = [i for i in range(10000)]
    plt.plot(epoch_list, sse_list)
    plt.title("Learning curve (Epoch vs SSE)")
    plt.xlabel("Epoch (iterations)")
    plt.ylabel("Sum of Squared Errors (SSE)")
    plt.show()
    return


if __name__ == '__main__':
    main()
