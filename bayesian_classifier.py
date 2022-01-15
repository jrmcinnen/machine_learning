"""
Bayesian classifier for cifar-10 data
"""

"""
Author:
Jere Mäkinen
Email:
jeremakinen98@gmail.com
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from skimage.transform import resize


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


# Resize the image to given size (1x1 by default)
def cifar10_color(x, size=1):
    xp = resize(x/255, (size, size))
    xp = xp.reshape((xp.size,))
    return xp


"""
Cifar-10 - Naive Bayesian mean and variance calculator

Function computes means and variances for all classes

Input: 
X = training data
Y = training data labels

Output:
mean = mean values for all classes
variances = variance values for all classes
"""
def cifar10_naivebayes_learn(X, Y):
    mean = np.zeros((10, 3))
    variance = np.zeros((10, 3))
    for i in range(10):
        mean[i] = np.mean(X[Y == i], axis=0)
        variance[i] = np.var(X[Y == i], axis=0)
    return mean, variance

"""
CIFAR-10 - Naive Bayesian classifier
Function finds a suitable label for input vector x based the naive bayesian 
formula

Input: 
x = vector to be labeled
mu = vector containing the mean values for each class
sigma = vector containg the variance values for each class
p = prior probabilities for each class

Output:
label = 0,1,2,...,9

Info:
In the formula the denominator is identical for each class so it is enough to
find the maximum value of the numerator
"""
def cifar10_classifier_naivebayes(x, mu, sigma, p=0.1*np.ones(10)):
    p_classes = np.zeros(10)
    for i in range(10):
        p_classes[i] = norm.pdf(x[0],loc=mu[i,0],
                        scale=np.sqrt(sigma[i,0]))*norm.pdf(x[1],loc=mu[i,1],
                        scale=np.sqrt(sigma[i,1]))*norm.pdf(x[2],loc=mu[i,2],
                        scale=np.sqrt(sigma[i,2]))*p[i]
    result = np.where(p_classes == np.max(p_classes))
    return result[0][0]



"""
Cifar-10 - Bayesian mean and covariance calculator

Function computes means and covariance matrices for all classes

Input: 
X = training data
Y = training data labels

Output:
mean = mean values for all classes
variances = variance values for all classes
"""
def cifar10_bayes_learn(X,Y):
    n = int(X.shape[1])
    mean = np.zeros((10,n))
    covariance = np.zeros((10,n,n))
    for i in range(10):
        mean[i] = np.mean(X[Y == i], axis=0)
        covariance[i] = np.cov(X[Y == i].T)
    return mean, covariance


"""
CIFAR-10 - Bayesian classifier
Function finds a suitable label for input vector x based the bayesian 
formula

Input: 
x = vector to be labeled
mu = vector containing the mean values for each class
sigma = vector containg the covariance matrices for each class
p = prior probabilities for each class

Output:
label = 0,1,2,...,9

Info:
In the formula the denominator is identical for each class so it is enough to
find the maximum value of the numerator
"""
def cifar10_classifier_bayes(x, mu, sigma, p=0.1*np.ones(10)):
    p_classes = np.zeros(10)
    for i in range(10):
        dist = multivariate_normal(mu[i],sigma[i])
        p_classes[i] = dist.pdf(x)*p[i]
    result = np.where(p_classes == np.max(p_classes))
    return result[0][0]

# Load the testing data
datadict = unpickle('cifar-10-python/test_batch')
X_test = datadict["data"]
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

X_training = np.concatenate((X1, X2, X3, X4, X5), axis=0)
Y_training = np.concatenate((Y1, Y2, Y3, Y4, Y5), axis=0)

# Convert vector back to images
X_training= X_training.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype('uint64')
# Resized training data to 1x1 images
X_training_1x1 = np.zeros((50000, 3))
for i in range(50000):
    X_training_1x1[i] = cifar10_color(X_training[i],1)

# Convert vectors back to images
X_test= X_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('uint64')
# Resized training data to 1x1 images
X_test_1x1 = np.zeros((10000, 3))
for i in range(10000):
    X_test_1x1[i] = cifar10_color(X_test[i])
    
print("Data converted to 1x1 images. \n")

# Compute means and variances
mu, sigma = cifar10_naivebayes_learn(X_training_1x1, Y_training)

# Computing the prior probabilities
p = np.zeros((10,1))
for i in range(10):
    p[i] = len(Y_training[Y_training == i])/len(Y_training)


test_naive_labels = np.zeros(X_test_1x1.shape[0])
for i in range(X_test_1x1.shape[0]):
    test_naive_labels[i] = cifar10_classifier_naivebayes(X_test_1x1[i], mu, sigma,p)


test_naive_acc = class_acc(test_naive_labels, Y_test)
print("Testing the naive bayesian classifier's accuracy on the test data:")
print(f'Accuracy is {test_naive_acc}\n')

# Compute means and covariance matrices
mu, sigma = cifar10_bayes_learn(X_training_1x1, Y_training)

test_bayes_labels = np.zeros(X_test_1x1.shape[0])
for i in range(X_test_1x1.shape[0]):
    test_bayes_labels[i] = cifar10_classifier_bayes(X_test_1x1[i], mu, sigma,p)

# The bayesian classifier gives better results than the naive version because
# it treats the three layers as a one unit rather the three seperate ones 
# (like the naive version does). The collaborative result of the three 
# different layers is the main idea of the rgb-images so it makes sense, which
# is something that the naive approach doesn-t take into account when handling
# all the layers separately. 
test_bayes_acc = class_acc(test_bayes_labels, Y_test)
print("Testing the bayesian classifier's accuracy on the test data with 1x1 images:")
print(f'Accuracy is {test_bayes_acc}\n')


# test different sized windows
# initialize containers
x = np.zeros(8)
y = np.zeros(8)
y[0] = test_bayes_acc
x[0] = 1

# image sizes to test
# stop at 16 because computing is getting too time consuming and inaccurate
size = [2,4,6,8,10,12,16]

for i in range(len(size)):
    x[i+1] = size[i]
    training_data = np.zeros((50000, int(3*size[i]**2)))
    for j in range(50000):
        training_data[j] = cifar10_color(X_training[j],size[i])
    test_data = np.zeros((10000, int(3*size[i]**2)))
    for j in range(10000):
        test_data[j] = cifar10_color(X_test[j],size[i])
    print(f"Data converted to {size[i]}x{size[i]} images.\n")
    mu, sigma = cifar10_bayes_learn(training_data, Y_training)
    test_labels = np.zeros(test_data.shape[0])
    for j in range(test_data.shape[0]):
        test_labels[j] = cifar10_classifier_bayes(test_data[j], mu, sigma,p)
    y[i+1] = class_acc(test_labels, Y_test)
    print("Testing the bayesian classifier's accuracy on the test data with" +
          f" {size[i]}x{size[i]} images:")
    print(f'Accuracy is {y[i+1]}\n')

# naive classifiers accuracy
plt.scatter(1,test_naive_acc,marker='^',label='Naive bayesian classifier')
# plot the classification accuracies of the different sized windows
plt.plot(x,y,'o',label='Bayesian classifier')
plt.legend()
plt.show()

