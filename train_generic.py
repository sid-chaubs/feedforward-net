from math import exp
from data import load_synth
import matplotlib.pyplot as plt
import numpy as np
from network.GenericNetwork import GenericNetwork
from random import seed

NUM_TRAIN = 60000
NUM_VALIDATION = 10000

seed(10)
(x_train, y_train), (x_validation, y_validation),g = load_synth(num_train = NUM_TRAIN, num_val = NUM_VALIDATION)

# normalize
x_validation = (x_validation - np.mean(x_train)) / np.std(x_train)
x_train = (x_train - np.mean(x_train)) / np.std(x_train)

alpha = 0.00001
epochs = 30

training_losses = []
training_accuracies = []

train_loss = 0
train_count = 0

validation_loss = 0
validation_count = 0

network = GenericNetwork()

for j in range(NUM_VALIDATION):
    X = x_train[j].tolist()
    Y = y_train[j]

    T = [0, 0]
    T[Y] = 1

    P, L = network.forward(X, T)
    train_loss += L

    if P.index(max(P)) == Y:
        train_count += 1

training_losses.append(train_loss / NUM_VALIDATION)
training_accuracies.append(train_count / NUM_VALIDATION)

validation_losses = []
validation_accuracies = []

for k in range(NUM_VALIDATION):
    X = x_validation[k].tolist()
    Y = y_validation[k]
    T = [0,0]
    T[Y] = 1
    P, L = network.forward(X, T)

    validation_loss += L
    if P.index(max(P)) == Y:
        validation_count += 1

validation_losses.append(validation_loss / NUM_VALIDATION)
validation_accuracies.append(validation_count / NUM_VALIDATION)

for e in range(epochs):
    # train the network
    for X, Y in zip(x_train, y_train):
        T = [0, 0]
        T[Y] = 1

        X = X.tolist()
        P, L = network.forward(X, T)
        dLdW, dLdB, dLdV, dLdC = network.backward(X, P, T)
        network.step(dLdW, dLdB, dLdV, dLdC, alpha)

    #training loss and accuracy
    training_loss = 0
    training_accuracy_count = 0

    for j in range(NUM_TRAIN):
        X = x_train[j].tolist()
        Y = y_train[j]
        T = [0, 0]
        T[Y] = 1
        P, L = network.forward(X, T)
        training_loss += L

        if P.index(max(P)) == Y:
            training_accuracy_count += 1

    training_losses.append(training_loss / NUM_TRAIN)
    training_accuracies.append(training_accuracy_count / NUM_TRAIN)

    #validation set loss and accuracy
    validation_loss = 0
    validation_accurate_count = 0

    for k in range(NUM_VALIDATION):
        X = x_validation[k].tolist()
        Y = y_validation[k]
        T = [0,0]
        T[Y] = 1
        P, L = network.forward(X, T)

        validation_loss += L
        if P.index(max(P)) == Y:
            validation_accurate_count += 1

    validation_losses.append(validation_loss / NUM_VALIDATION)
    validation_accuracies.append(validation_accurate_count / NUM_VALIDATION)

plt.plot([x for x in validation_losses], label = 'Validation loss')
plt.plot([x for x in training_losses], label = 'Training loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
