Readme for the MLP-DT implementation
-------------------------------------

Authors: Akshay Karki and Krishna Tippur Gururaj

NOTE: The python package "beautifultable" needs to be installed prior to executing the supplied scripts.
The implementation is done on a Python 3.6 compiler.

Multi-layer perceptron:

1. The model is to be trained using trainMLP.py. This script is to be supplied with the training data CSV file. The CSV file is to represent the data samples of two attributes.
Usage: python3 trainMLP.py train_data.csv

Running this will generate network weights for epochs 0, 10, 100, 1000, and 10000. These are put in files called MLPweights0.csv, MLPweights10.csv, MLPweights100.csv, MLPweights1000.csv, and MLPweights10000.csv respectively. It will also plot a curve of SSE vs epoch (to show the learning curve of the training model).

These files are then to be used as inputs to the executeMLP.py along with a CSV file containing test data.
Usage: python3 executeMLP.py MLPweights10000.csv test_data.csv

The executeMLP.py will generate a confusion matrix, profit, accuracy, and mean per class accuracy on stdout. It will also plot the decision boundary along with a plot of the test data points on them (to showcase the accuracy and efficiency of the model).

IMPORTANT: The number of nodes in the hidden layer can be configured by just changing one parameter in each of the two scripts. There is a parameter at the beginning of trainMLP and executeMLP scripts called "hn_count". This needs to be kept at the same value in both!

Decision Tree:

Running the trainDT.py will ask the user for a training data to train the model on
Usage: python3 trainDT.py train_data.csv

The trainDT.py will generate the entire root (with indentation showing child relationship) and tree metrics before and after pruning the tree. It also generates 2 decision plots and plots the training data samples on them.

It also generates 2 data files, DT_before.py and DT_after.py. These files are the decision trees created by the trainDT.py file. These are to be supplied to the executeDT.py file.
The executeDT.py file takes in 2 arguments. The first argument is the data file containing the decision tree and the second argument is the testing data

"USAGE: python3 executeDT <DT_file> <data_file>"
"<DT_file> = decision tree file created by trainDT.py"
"<data_file> = testing data file"

The executeDT.py will generate a confusion matrix, profit, accuracy, and mean per class accuracy on stdout. It will also plot the decision boundary along with a plot of the test data points on them.