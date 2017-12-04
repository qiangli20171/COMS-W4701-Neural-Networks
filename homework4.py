#!/usr/bin/env python3

####################################################################
##
##  COMS W4701 Artificial Intelligence
##  Columbia University
##  
##  Homework 4: Neural Networks
##
##  UNI: srb2208
##
####################################################################

'''
A short conversation on binary classifiers vs. categorical classifiers:

From my testing with this assignment, I would argue that binary classification is easier than classification into
10 categories. My reasoning for this is two fold. First, a binary classifier starts off with a baseline accuracy of 
50%. Any worse and one can simply invert the the classifier and it will be better than 50%. As a result, it is much 
easier to build accuracy levels up. Further, because there is are less outcomes, the models with the same number of
parameters will learn more features for each binary option than they would for each of 10 categories. This further
enhances the accuracy of the model. Finally the binary classification model I used had one convolution layer, one 
pooling layer, one dropout layer, one dense layer, and one output layer. It was the first combination I tried and
proved to be quite accurate (>90%) with no tuning of options or anything. I am confident that with a little more 
work, the classifier could prove even better. Regardless it was quite easy to get this outcome and only took 20 
minutes to train on my machine. For comparison, the baseline convolutional network given in part three of this 
assignment was taking ~10 minutes an epoch on my machine. Because the classifier has less outputs to worry about, 
was faster to develop for a high accuracy, and faster to train to a high accuracy, it is easier to create a binary
classifier than it is to create a 10 category classifier.

'''

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers

# Load in the CIFAR-10 Data, normalize the inputs and 1 hot the outputs
def load_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test

    # Normalize the training and test data
    xtrain_norm = np.multiply(xtrain, 1/255)
    xtest_norm = np.multiply(xtest, 1/255)

    # Declare varibles to hold the 1 hot output labels
    ytrain_1hot = []
    ytest_1hot = []

    # For each label, change it from a number to 1 hot representation
    for label in ytrain:

        # Create a 10x1 zero array and add the 1 hot
        hot = np.zeros(shape=(10,), dtype = int)
        hot[label] = 1

        # Add it to the 1 hot array
        ytrain_1hot.append(hot)

    for label in ytest:

        # Create a 10x1 zero array and add the 1 hot
        hot = np.zeros(shape=(10,), dtype = int)
        hot[label] = 1

        # Add it to the 1 hot array
        ytest_1hot.append(hot)

    # Save some memory before continuing
    del xtrain
    del xtest
    del ytrain
    del ytest

    # Return the data
    return xtrain_norm, np.array(ytrain_1hot), xtest_norm, np.array(ytest_1hot)

# Build the neural network
# Output of evaluate(xtest, ytest_1hot):
# 10000/10000 [==============================] - 5s 488us/step
# [1.5083699712753296, 0.46129999999999999]
def build_multilayer_nn():

    # Create the neural network
    nn = Sequential()

    # Create a layer to preprocess the input - this layer flattens the 32x32x3 array to be usable by the hidden layer
    flatten = Flatten(input_shape=(32, 32, 3))
    nn.add(flatten)

    # Create the hidden layer and add it to the network
    # - This creates a 100 neuron layer, fully connected to the inputs with the rectifier activation function
    hidden = Dense(units=100, activation="relu")
    nn.add(hidden)

    # Create the output layer and add it to the network
    # - Like the hidden layer this layer will be dense (fully connected)
    # - This layer will have 10 outputs, one for each classification
    # - The activation function is a softmax which makes it so the 10 output values add up to 1.0
    # - This output activation function is similar to a probability distribution
    output = Dense(units=10, activation="softmax")
    nn.add(output)

    # Return the network
    return nn

# Train the neural network
def train_multilayer_nn(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    model.fit(xtrain, ytrain, epochs=20, batch_size=32)        
 

# Build a convolutional neural network
# This network will have the following:
# - 2 convolutional layers with 32 feature maps and filter size 3x3
# - 1 pooling layer to reduce the feature maps to 16x16
# - 1 dropout layer which drops 25% of the units to prevent overfitting
# - 2 convolutional layers with 32 feature maps of filter size 3x3
# - 1 pooling layer to reduce the feature maps to 8x8
# - 1 dropout layer which drops 50% of the units to prevent overfitting
# - 1 dense layer with 250 neurons
# - 1 dense layer with 100 neurons
# - 1 output layer which reduces the output to a 10 value array similar to a probability distribution
def build_convolution_nn():

    # Create the neural network
    nn = Sequential()

    # Create the first convolution layers and add it twice to the neural network
    conv_in = Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3))
    conv = Conv2D(32, (3, 3), activation='relu', padding="same")
    nn.add(conv_in)
    nn.add(conv)

    # Create the first pooling layer to reduce the feature maps to 16x16
    pool = MaxPooling2D(pool_size=(2, 2))
    nn.add(pool)

    # Create and add the first dropout layer
    drop1 = Dropout(0.25)
    nn.add(drop1)

    # Add the second convolutional layers to the network
    nn.add(conv)
    nn.add(conv)

    # Add the second pooling layer to reduce the feature maps to 8x8
    nn.add(pool)

    # Create and add the second dropout layer
    drop2 = Dropout(0.5)
    nn.add(drop2)

    # Create a layer to flatten the data - this layer flattens the 8x8x32 array to be usable by the hidden layer
    flatten = Flatten()
    nn.add(flatten)

    # Create the first dense hidden layer and add it to the network
    # - This creates a 250 neuron layer, fully connected to the inputs with the rectifier activation function
    hidden1 = Dense(units=250, activation="relu")
    nn.add(hidden1)

    # Create the second dense hidden layer and add it to the network
    # - This creates a 100 neuron layer, fully connected to the inputs with the rectifier activation function
    hidden2 = Dense(units=100, activation="relu")
    nn.add(hidden2)

    # Create the output layer and add it to the network
    # - Like the hidden layer this layer will be dense (fully connected)
    # - This layer will have 10 outputs, one for each classification
    # - The activation function is a softmax which makes it so the 10 output values add up to 1.0
    # - This output activation function is similar to a probability distribution
    output = Dense(units=10, activation="softmax")
    nn.add(output)

    # Return the network
    return nn
    
# Train the convolutional neural network
def train_convolution_nn(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    model.fit(xtrain, ytrain, epochs=20, batch_size=32)
    
# Load the CIFAR-10 Data and modify the outputs to be binary for animal or vehical
def get_binary_cifar10():    
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test

    # Normalize the training and test data
    xtrain_norm = np.multiply(xtrain, 1/255)
    xtest_norm = np.multiply(xtest, 1/255)

    # Modify ytrain and ytest to be for binary classification between a vehical and animal
    # - 1 => Animal
    # - 0 => Vehicle
    ytrain = [1 if label > 1 and label < 8 else 0 for label in ytrain]
    ytest = [1 if label > 1 and label < 8 else 0 for label in ytest]    

    # Save some memory before continuing
    del xtrain
    del xtest

    # Return the data
    return xtrain_norm, np.reshape(ytrain, (50000,)), xtest_norm, np.reshape(ytest, (10000,))

# Build the neural network
# Output of evaluate(xtest, ytest):
# 10000/10000 [==============================] - 30s 3ms/step
# [0.19405419319868089, 0.91920000000000002]
def build_binary_classifier(): 

    # Create the neural network
    nn = Sequential()

    # Create a convolution layers and add it to the neural network
    conv_in = Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3))
    nn.add(conv_in)

    # Create a pooling layer to reduce the feature maps to 16x16
    pool = MaxPooling2D(pool_size=(2, 2))
    nn.add(pool)

    # Create and add a dropout layer
    drop1 = Dropout(0.25)
    nn.add(drop1)

    # Create a layer to flatten the data - this layer flattens the 16x16x32 array to be usable by the hidden layer
    flatten = Flatten()
    nn.add(flatten)

    # Create a dense hidden layer and add it to the network
    # - This creates a 10 neuron layer, fully connected to the inputs with the rectifier activation function
    hidden = Dense(units=10, activation="relu")
    nn.add(hidden)

    # Create the output layer and add it to the network
    # - Like the hidden layer this layer will be dense (fully connected)
    # - This layer will have 1 output
    # - The activation function is a sigmoid
    output = Dense(units=1, activation="sigmoid")
    nn.add(output)

    # Return the network
    return nn

# Train the binary classifier neural network
def train_binary_classifier(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    model.fit(xtrain, ytrain, epochs=20, batch_size=32)


if __name__ == "__main__":

    # Write any code for testing and evaluation in this main section.
    
    '''
    xtrain, ytrain_1hot, xtest, ytest_1hot = load_cifar10()
    nn = build_convolution_nn()
    train_convolution_nn(nn, xtrain, ytrain_1hot)
    print('--')
    out = nn.evaluate(xtest, ytest_1hot)
    print(out)
    '''

    xtrain, ytrain, xtest, ytest = get_binary_cifar10()
    nn = build_binary_classifier()
    train_binary_classifier(nn, xtrain, ytrain)
    print('--')
    out = nn.evaluate(xtest, ytest)
    print(out)


