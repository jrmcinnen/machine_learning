"""
1-nn classifier for cifar-10 data

Also includes random classifier and class accuracy tester.
"""

"""
Author:
Jere Mäkinen
Email:
jeremakinen98@gmail.com
"""

import pickle
import numpy as np
from random import randint


# Function to help with reading the data 
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


"""
CIFAR-10 - Evaluation

Function calculates the classification accuracy by comparing the predicted
values to the ground truth.

Input:
pred = predicted labels,
gt = ground truth labels

Output:
acc (in [0, 1.0]) = classification accuracy

Info:
pred and gt should be same size (n times 1 vectors)
"""
def class_acc(pred, gt):
    n = len(pred)
    # counter for values that are different
    miss = 0
    for i in range(n):
        # if values at index i are different add 1 to miss-counter
        if pred[i] != gt[i]:
            miss += 1
    # compute the portion of "missed" labels
    acc = (n-miss)/n
    return acc


"""
CIFAR-10 - Random clasifier

Function gives a random label for the given vector

Input:
x = vector to be labeled

Output:
label = 0,1,2,...,9 randomly assigned label
"""
def cifar10_classifier_random(x):
    # assigning random label by utilizing random-library
    label = randint(0, 9)
    return label


"""
CIFAR-10 - 1-NN classifier

Function finds the match for the input vector x in the training data set. The
label from the closest match is assigned to vector x.

Input:
x = vector to be labeled
trdata = training data
trlabels = training data labels

Output:
label = 0,1,2,...,9
"""
def cifar10_classifier_1nn(x, trdata, trlabels):
    n = len(trdata)
    # Fill a matrix with input vector x
    x_matrix = np.tile(x,(n,1))
    # Cmpute the euclidean lenght to each  of the training data point
    errors = (x_matrix-trdata)**2
    errors = errors.sum(axis=1)
    # Finding the smallest error
    result = np.where(errors == np.min(errors))
    return  trlabels[result[0][0]]


# Load the testing data
datadict = unpickle('cifar-10-python/test_batch')
X_test = datadict["data"]
# Convert int8 values to int32
X_test = X_test.astype(np.uint32)
Y_test = datadict["labels"]

labeldict = unpickle('cifar-10-python/batches.meta')
label_names = labeldict["label_names"]

# Load the training data
datadict1 = unpickle('cifar-10-python/data_batch_1')
X1 = datadict1["data"]
Y1 = datadict1["labels"]
datadict2 = unpickle('cifar-10-python/data_batch_2')
X2 = datadict2["data"]
Y2 = datadict2["labels"]
datadict3 = unpickle('cifar-10-python/data_batch_3')
X3 = datadict3["data"]
Y3 = datadict3["labels"]
datadict4 = unpickle('cifar-10-python/data_batch_4')
X4 = datadict4["data"]
Y4 = datadict4["labels"]
datadict5 = unpickle('cifar-10-python/data_batch_5')
X5 = datadict5["data"]
Y5 = datadict5["labels"]

X_training = np.concatenate((X1,X2,X3,X4,X5), axis=0)
# Convert int8 values to int32
X_training = X_training.astype(np.uint32)
Y_training = np.concatenate((Y1,Y2,Y3,Y4,Y5), axis=0)


# Test the class_acc -function with test values
test_acc = class_acc(Y_test, Y_test)
print('Testing the classification accuracy function with two identical ' +
      'label sets (should be exactly 1.0):')
print(f'Accuracy is {test_acc}\n')

# Test the random classifier
random_labels = np.zeros(X_test.shape[0])
for i in range(len(random_labels)):
    random_labels[i] = cifar10_classifier_random(X_test[i:])

random_acc = class_acc(random_labels, Y_test)
print("Testing the random classifier's accuracy (should be around 0.1):")
print(f'Accuracy is {random_acc}\n')

# Test the 1-NN classifier on the training data
training_1nn_labels = np.zeros(X1.shape[0])
for i in range(X1.shape[0]):
    training_1nn_labels[i] = cifar10_classifier_1nn(X1[i,:],X_training,
                                                    Y_training)
    
test_1nn_training_acc = class_acc(training_1nn_labels, Y_training)

print("Testing the 1-NN classifier's functionality with a part of the " 
      "training data (should be exactly 1.0):")
print(f'Accuracy is {test_1nn_training_acc}\n')

# Test the 1-NN classifier on the test date
test_1nn_labels = np.zeros(X_test.shape[0])
for i in range(X_test.shape[0]):
    test_1nn_labels[i] = cifar10_classifier_1nn(X_test[i,:],X_training,
                                                Y_training)
    
test_1nn_acc = class_acc(test_1nn_labels, Y_test)
print("Testing the 1-NN classifier's accuracy on the test data:")
print(f'Accuracy is {test_1nn_acc}\n')
