"""
Neural classifier for cifar-10 data
"""

"""
Author:
Jere Mäkinen
Email:
jeremakinen98@gmail.com
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras


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
Converts labels 0,1,...,9 to onehot vectors

Input:
Y = set of labels

Output:
Y_onehot = set of onehot vectors
"""
def labels_to_onehot(Y):
    Y_onehot = np.zeros((len(Y),10))
    for i in range(10):
        Y_onehot[np.where(Y == i),i] = 1
    return Y_onehot

"""
Converts onehot vectors to labels 0,1,...,9

Input:
Y_onehot = set of onehot vectors

Output:
Y = set of labels
"""
def onehot_to_label(Y_onehot):
    Y = np.zeros(Y_onehot.shape[0])
    for i in range(len(Y)):
        Y[i] = np.where(Y_onehot[i] == 1)[0]
    return Y

# Load the testing data
datadict = unpickle('cifar-10-python/test_batch')
X_test = datadict["data"]
# Convert int8 values to int32
X_test = X_test.astype(np.uint32)
# Scale the data
X_test = X_test*1/255
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
# Scale the data
X_training = X_training*1/255
Y_training = np.concatenate((Y1,Y2,Y3,Y4,Y5), axis=0)

Y_training_onehot = labels_to_onehot(Y_training)
Y_lol = onehot_to_label(Y_training_onehot)


"""
Neural classifier
"""

# Initialize sequential model
model = Sequential()

# 100 layers
model.add(Dense(100, input_dim=3072, activation='sigmoid'))
# output 10 layers
model.add(Dense(10, activation='sigmoid'))
# learning rate 0.1
opt = keras.optimizers.SGD(lr=0.1)
model.compile(optimizer=opt, loss='categorical_crossentropy')
# number of epochs
num_of_epochs = 100
# train the model
tr_hist = model.fit(X_training, Y_training_onehot, epochs=num_of_epochs, verbose=1)
# plot how the loss develops as the epochs go
plt.plot(tr_hist.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')

pred_labels_training = np.zeros(len(Y_training))
pred_labels_training_2 = np.squeeze(model.predict(X_training))
for i in range(pred_labels_training_2.shape[0]):
    pred_labels_training[i] = np.where(pred_labels_training_2[i] == np.max(pred_labels_training_2[i]))[0][0]

pred_acc = class_acc(pred_labels_training, Y_training)

print("\nTesting accuracy of the neural classifier:")
print(f"Accuracy for training data is {pred_acc}")

pred_labels_test = np.zeros(len(Y_test))
pred_labels_test_2 = np.squeeze(model.predict(X_test))
for i in range(pred_labels_test_2.shape[0]):
    pred_labels_test[i] = np.where(pred_labels_test_2[i] == np.max(pred_labels_test_2[i]))[0][0]
    
pred_acc = class_acc(pred_labels_test, Y_test)

print(f"Accuracy for test data is {pred_acc}\n")
