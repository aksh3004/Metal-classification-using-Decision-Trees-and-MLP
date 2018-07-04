import math
import operator
import matplotlib.pyplot as plt
import sys

class DataPoint:
    """
    Class of each data point in the given csv file, with 3 attributes
    """
    def __init__(self, id, attrib1, attrib2, class_value):
        self.id = id
        self.x = attrib1
        self.y = attrib2
        self.z = class_value
        self.threshold = 0
        self.count_class = [0, 0, 0, 0]
        self.right_list = []
        self.left_list= []
        self.attribnumber = 0
        self.rightnode = None
        self.leftnode = None
        self.is_leafnode = False
        self.leafvalue = 0


def readInput(filename):
    """
    Reads the file and creates a list of objects
    :param filename: the file to be read
    :return: the list of objects
    """
    file = open(filename, 'r')
    data = file.readlines()
    point_list = []
    for line in range(len(data)):
        store = data[line].strip().split(',')
        # check for empty spaces or new line characters
        if len(store) != 1:
            datapoint = DataPoint(line, float(store[0]), float(store[1]), int(store[2]))
            point_list.append(datapoint)
    return point_list


def count_class(point_list):
    """
    Compute the count of each class in a given sample
    :param point_list: the list of samples
    :return: the list of counts of each class
    """
    all_counts = [0, 0, 0, 0]
    for index in range(len(point_list)):
        all_counts[point_list[index].z - 1] += 1
    return all_counts


def find_entropy_child(total_count, child):
    """
    Compute the entropy of given subtree
    :param total_count: the total count of objects in the subtree
    :param child: the side of subtree, i.e., right or left
    :return: the computed entropy
    """
    compute = 0
    all_counts = [0, 0, 0, 0]
    # iterate over all objects in the subtree and count number of classes
    for element in range(len(child)):
        value = child[element].z
        if value == 1:
            all_counts[0] += 1
        elif value == 2:
            all_counts[1] += 1
        elif value == 3:
            all_counts[2] += 1
        else:
            all_counts[3] += 1

    for index in range(len(all_counts)):
        if all_counts[index] != 0:
            compute += all_counts[index] / total_count * math.log(all_counts[index] / total_count, 2)

    # compute the entropy of all 4 classes combined
    entropy_split = -1 * compute
    return entropy_split


def find_entropy_parent(total_count, all_counts):
    """
    Find the entropy of the parent
    :param total_count: the total points in the list
    :param all_counts: the list of class counts for each sample
    :return: the computed entropy
    """
    compute = 0
    for index in range(len(all_counts)):
        if all_counts[index] != 0:
            compute += all_counts[index] / total_count * math.log(all_counts[index] / total_count, 2)

    entropy_unsplit = -1 * compute
    return entropy_unsplit


def find_info_gain(point_list, total_count, entropy_unsplit):
    """
    Function to compute information gain
    :param point_list: the list of samples
    :param total_count: the total count of points in the list
    :param entropy_unsplit: the entropy of the parent
    :return: the list of characteristics of given sample
    """
    first_list = sorted(point_list, key=lambda elem: elem.x)        # sort the list of first attribute instances
    second_list = sorted(point_list, key=lambda elem: elem.y)       # sort the list of second attribute instances

    value1 = computations(first_list, total_count, entropy_unsplit, attrib=1)
    value2 = computations(second_list, total_count, entropy_unsplit, attrib=2)

    # check which attribute has a higher information gain and return that attribute features
    if value1[0] > value2[0]:
        for _ in point_list:
            if _.id == value1[2]:
                _.attribnumber = 1                                  # set the attribute number of the sample to 1
                return value1
    else:
        for _ in point_list:
            if _.id == value2[2]:
                _.attribnumber = 2                                  # set the attribute number of the sample to 2
                return value2


def build_tree(point_list, root=None, depth=0, check=0):
    """
    Function to build the tree
    :param point_list: the list of elements to be stored in the tree
    :param root: the root node
    :param depth: the depth of the tree
    :param check: a flag to check whether the recursive call came from the left subtree or right subtree
    :return: None
    """
    all_counts = [0, 0, 0, 0]
    total_count = len(point_list)
    # iterate over the entire list of points and check the class count for each sample
    for _ in range(len(point_list)):
        if point_list[_].z == 1:
            all_counts[0] += 1
        elif point_list[_].z == 2:
            all_counts[1] += 1
        elif point_list[_].z == 3:
            all_counts[2] += 1
        else:
            all_counts[3] += 1

    # check if all the values in the list belong to the same class
    if len(point_list) == all_counts[0] or len(point_list) == all_counts[1] or len(point_list) == all_counts[2] or len(point_list) == all_counts[3]:
        # create a new leaf node with the root features except the leaf value
        leaf = DataPoint(root.id, root.x, root.y, root.z)
        leaf.is_leafnode = True
        leaf.count_class = root.count_class
        leaf.attribnumber = 0
        leaf.threshold = root.threshold
        # check whether to set the new leaf node as the right child or the left child
        if check == 0:
            root.rightnode = leaf
        else:
            root.leftnode = leaf

        # check which class all points in the list belong to
        if len(point_list) == all_counts[0]:
            leaf.leafvalue = 1
            return all_counts[0], leaf

        elif len(point_list) == all_counts[1]:
            leaf.leafvalue = 2
            return all_counts[1], leaf

        elif len(point_list) == all_counts[2]:
            leaf.leafvalue = 3
            return all_counts[2], leaf

        elif len(point_list) == all_counts[3]:
            leaf.leafvalue = 4
            return all_counts[3], leaf

    else:
        entropy_unsplit = find_entropy_parent(total_count, all_counts)
        value = find_info_gain(point_list, total_count, entropy_unsplit)
        id_of_element = value[2]
        for elem in point_list:
            if elem.id == id_of_element:
                root = elem
        root.threshold = value[1]
        root.right_list = value[3]
        root.left_list = value[4]

        _, root.rightnode = build_tree(root.right_list, root, depth + 1, check=0)
        _, root.leftnode = build_tree(root.left_list, root, depth + 1, check=1)
        return all_counts, root


def computations(list_sorted, total_count, entropy_unsplit, attrib):
    """
    Function to perform all computations
    :param list_sorted: the sorted list of attribute values
    :param total_count: the total count of points in the list
    :param entropy_unsplit: the entropy of the parent
    :param attrib: a flag to check whether the first attribute has a better IG value than the second attribute
    :return: the list of sample characteristics
    """
    mid_value = 0
    mid_and_IG = []
    # iterate over the entire list of sorted elements
    for index in range(len(list_sorted)):
        list_sorted[index].right_list, list_sorted[index].left_list = [], []
        if index != len(list_sorted) - 1:

            if attrib == 1:
                mid_value = (list_sorted[index].x + list_sorted[index + 1].x) / 2   # calculate the middle value of adjacent instances
                # iterate over the entire list and check which instances are above and below the middle value
                # if they are above the middle value, append them to the right child list, else append them to the left child list
                for inner_index in range(len(list_sorted)):
                    if list_sorted[inner_index].x >= mid_value:
                        list_sorted[index].right_list.append(list_sorted[inner_index])
                    else:
                        list_sorted[index].left_list.append(list_sorted[inner_index])
            else:
                mid_value = (list_sorted[index].y + list_sorted[index + 1].y) / 2   # calculate the middle value of adjacent instances
                # iterate over the entire list and check which instances are above and below the middle value
                # if they are above the middle value, append them to the right child list, else append them to the left child list
                for inner_index in range(len(list_sorted)):
                    if list_sorted[inner_index].y >= mid_value:
                        list_sorted[index].right_list.append(list_sorted[inner_index])
                    else:
                        list_sorted[index].left_list.append(list_sorted[inner_index])
        list_sorted[index].count_class = count_class(list_sorted)

        # store the total count of points or objects in the right node list and in the left node list
        total_count_right = len(list_sorted[index].right_list)
        total_count_left = len(list_sorted[index].left_list)
        if total_count_right == 0 or total_count_left == 0:
            continue

        entropy_right = find_entropy_child(total_count_right, list_sorted[index].right_list)         # compute the entropy of right child
        entropy_left = find_entropy_child(total_count_left, list_sorted[index].left_list)            # compute the entropy of left child

        # calculate the information gain as the difference in entropy of parent and entropy of both children
        IG = entropy_unsplit - ((total_count_right / total_count) * entropy_right + (total_count_left / total_count) * entropy_left)

        # append to a list the IG value and the middle value
        mid_and_IG.append([IG, mid_value, list_sorted[index].id, list_sorted[index].right_list, list_sorted[index].left_list])

    mid_and_IG.sort(key=lambda x: x[0])                                         # sort the list based on their IG values
    index, value = max(enumerate(mid_and_IG), key=operator.itemgetter(0))       # store the index and value of max. IG value

    return value


def classify_samples(sample, root):
    """
    Classify given sample for the decision plot
    :param sample: the sample to classify
    :param root: the root of the tree
    :return: the leaf value
    """
    if root.attribnumber == 1:
        if sample[0] > root.threshold:
            return classify_samples(sample, root.rightnode)
        else:
            return classify_samples(sample, root.leftnode)
    elif root.attribnumber == 2:
        if sample[1] > root.threshold:
            return classify_samples(sample, root.rightnode)
        else:
            return classify_samples(sample, root.leftnode)
    else:
        value = root.leafvalue
        return value


def classify_full(sample_list, point_list, root):
    """
    Classify the list of points and plot them on the decision plot
    :param sample_list: the list of points to draw the region
    :param point_list: the list of training data points
    :param root: the root of the decision tree
    :return:
    """
    correct1x, correct1y, correct2x, correct2y, correct3x, correct3y, correct4x, correct4y = [], [], [], [], [], [], [], []
    for index in range(len(sample_list)):
        prediction = classify_samples(sample_list[index], root)
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
    #plt.show()


def chi_squared_test(object, significance):
    """
    Perform the chi-squared test on the given object to see whether splitting helps
    :param object: the object to be checked for significance
    :param significance: the significance value
    :return: True if the chi-squared value is lesser than the computed k value, False otherwise
    """
    in_left = sum(object.leftnode.count_class) / sum(object.count_class)        # the total count of each class in the left node
    in_right = sum(object.rightnode.count_class) / sum(object.count_class)      # the total count of each class in the right node
    expected_right, expected_left = [], []

    # iterate over the list of count values and compute the expected value in left and right nodes
    for _ in range(len(object.count_class)):
        expected_left.append(object.count_class[_] * in_left)
        expected_right.append(object.count_class[_] * in_right)

    k = 0
    # iterate over the list of count values to compute value of k
    for index in range(len(object.count_class)):
        if expected_left[index] != 0:
            k +=  math.pow(expected_left[index] - object.leftnode.count_class[index], 2) / expected_left[index]
        if expected_right[index] != 0:
            k += math.pow(expected_right[index] - object.rightnode.count_class[index], 2) / expected_right[index]
    # check for significance
    print(k)
    if significance == 0.01 and k < 11.345:
        return k, True
    if significance == 0.05 and k < 7.815:
        return k, True
    else:
        return k, False


def pruning(root, depth, node_depth, significance):
    """
    Function to prune the unwanted branches of the decision tree
    :param root: the root node of the tree
    :param significance: the significance level at which to test for pruning
    :return: None
    """
    # check whether the root is a leaf node or not

    if root is not None:
        if root.rightnode.is_leafnode and root.leftnode.is_leafnode:
                value_of_k, bool = chi_squared_test(root, significance)
                print(value_of_k)
                if bool:
                    root.leafvalue = max(root.rightnode.count_class, root.leftnode.count_class)
                    root.rightnode = None
                    root.leftnode = None
                    root.is_leafnode = True
                return
    else:
        if root is not None:
            if not root.leftnode.is_leafnode:
                pruning(root.leftnode, depth + 1, node_depth, significance)
            if not root.rightnode.is_leafnode:
                pruning(root.rightnode, depth + 1, node_depth, significance)


def write_to_file(pointer, root, depth=0):
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
    if not root.is_leafnode:
        # write the attribute number, threshold and leaf value if present
        line = str('\t' * depth) + str(root.attribnumber) + ',' + str(root.threshold) + ',' + str(0) + '\n'
    else:
        # compute the leaf value
        value = [x for x in range(len(root.count_class)) if root.count_class[x] != 0]
        line = str('\t' * depth) + str(root.attribnumber) + ',' + str(root.threshold) + ',' + str(root.leafvalue) + '\n'
        pointer.write(line)
        return

    pointer.write(line)
    # recursively call the left and right subtree
    write_to_file(pointer, root.leftnode, depth + 1)
    write_to_file(pointer, root.rightnode, depth + 1)


def find_maxdepth(root):
    """
    Find the max depth of the decision tree
    :param root: the root of the tree
    :return: the max depth
    """
    if root.leftnode is None and root.rightnode is None:
        return 0
    return 1 + max(find_mindepth(root.leftnode), find_mindepth(root.rightnode))


def find_mindepth(root):
    """
    Find the min depth of the tree
    :param root: the root of the decision tree
    :return: the min depth
    """
    if root.leftnode is None and root.rightnode is None:
        return 0
    return 1 + min(find_mindepth(root.leftnode), find_mindepth(root.rightnode))


def create_list_for_plotting():
    """
    Create the list of x and y values for plotting the graph
    :return: the list of x and y points
    """
    step_size = 0.001
    data = []
    for index in range(1000):
        index /= 1000
        for idx in range(1000):
            idx /= 1000
            data.append([index + step_size, idx + step_size])
    return data


def print_tree(root, depth=0):
    """
    Prints the decision tree given the root node
    :param root: the root node of the tree
    :param depth: the depth of the tree
    :return: None
    """
    if root is not None:
        if root.leafvalue == float(0):
            print('\t' * depth + 'Attribute number: ' + str(root.attribnumber) + ' Threshold value: ' + str(root.threshold) + ' Non-terminal node')
        else:
            print('\t' * depth + 'Attribute number: ' + str(root.attribnumber) + ' Threshold value: ' + str(root.threshold) + ' Class value: ' + str(root.leafvalue))
        print_tree(root.leftnode,depth+1)
        print_tree(root.rightnode,depth+1)


def main():
    """
    The main function
    :return: None
    """
    if len(sys.argv) < 2:
        print("input training file not supplied!")
        exit(0)
    filename = sys.argv[1]
    point_list = readInput(filename)

    print('-------------- Before pruning ------------------')
    print()
    count, root =  build_tree(point_list)
    print_tree(root)
    print()

    print('******** Depth of the tree *********')
    print()
    max_depth = find_maxdepth(root)
    min_depth = find_mindepth(root)
    print('Min depth: ', min_depth)
    print('Max depth: ', max_depth)
    print()

    data = create_list_for_plotting()
    classify_full(data, point_list, root)

    pointer = open('DT_before.txt', 'w')
    write_to_file(pointer, root)

    print('---------------- After pruning -----------------')
    print()
    pruning(root, 0, max_depth - 1, significance=0.01)

    print_tree(root)
    print()
    pointer = open('DT_after.txt','w')
    write_to_file(pointer, root)

    print('******** Depth of the tree *********')
    print()
    max_depth = find_maxdepth(root)
    min_depth = find_mindepth(root)
    print('Min depth: ', min_depth)
    print('Max depth: ', max_depth)
    print()

    data_pruned = create_list_for_plotting()
    classify_full(data_pruned, point_list, root)

    pointer.close()


if __name__ == '__main__':
    main()
