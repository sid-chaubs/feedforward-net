import matplotlib.pyplot as plt
import numpy as np
from data import load_mnist
from copy import deepcopy
from random import seed
from network.MNISTNetwork import MNISTNetwork

seed(111)

(x_train, T_train), (x_validation, T_validation), g = load_mnist(final = False)
(_, _), (x_test, T_test), _ = load_mnist(final = True)

# normalize
X_train = (x_train - np.mean(x_train))/np.std(x_train)
X_validation = (x_validation - np.mean(x_train)) / np.std(x_train)
X_test = (x_test - np.mean(x_train)) / np.std(x_train)

network = MNISTNetwork()
epochs = 10
alpha = 0.01

training_losses = list()
validation_losses = list()
test_losses = list()

training_accuracies = list()
validation_accuracies = list()
test_accuracies = list()

# losses and accuracies before training
training_count = 0
training_loss = 0
training_accurate_count = 0

for X, T in zip(X_train, T_train):
    P, L = network.forward(X, T)
    training_loss += L
    training_count += 1

    if np.argmax(P) == T:
        training_accurate_count += 1

# add mean training loss & accuracy to list
training_losses.append(training_loss / training_count)
training_accuracies.append(training_accurate_count / training_count)

validation_count = 0
validation_loss = 0
validation_accurate_count = 0
for X, T in zip(X_validation, T_validation):
    P, L = network.forward(X, T)
    validation_loss += L

    if np.argmax(P) == T:
        validation_accurate_count += 1

    validation_count += 1

# add mean validation loss & accuracy to list
validation_losses.append(validation_loss / validation_count)
validation_accuracies.append(validation_accurate_count / validation_count)

test_count = 0
test_loss = 0
test_accurate_count = 0
for X, T in zip(X_test, T_test):
    P, L = network.forward(X, T)
    test_loss += L

    if np.argmax(P) == T:
        test_accurate_count += 1

    test_count += 1

# add mean test loss & accuracy to list
test_losses.append(test_loss / test_count)
test_accuracies.append(test_accurate_count / test_count)

for epoch in range(epochs):
    training_count = 0
    training_loss = 0
    training_accurate_count = 0

    for X, T in zip(X_train, T_train):
        P, L = network.forward(X, T)
        dLdV, dLdC, dLdW, dLdB = network.backward(X, T, P)
        network.step(dLdV, dLdC, dLdW, dLdB, alpha)

    # training loss after first epoch
    for X, T in zip(X_train, T_train):
        P, L = network.forward(X, T)
        training_loss += L
        training_count += 1

        if np.argmax(P) == T:
            training_accurate_count += 1

    # add mean training loss & accuracy to list
    training_losses.append(training_loss / training_count)
    training_accuracies.append(training_accurate_count / training_count)

    validation_count = 0
    validation_loss = 0
    validation_accurate_count = 0
    for X, T in zip(X_validation, T_validation):
        P, L = network.forward(X,T)
        validation_loss += L

        if np.argmax(P) == T:
            validation_accurate_count += 1

        validation_count += 1

    # add mean validation loss & accuracy to list
    validation_losses.append(validation_loss / validation_count)
    validation_accuracies.append(validation_accurate_count / validation_count)

    test_count = 0
    test_loss = 0
    test_accurate_count = 0
    for X, T in zip(X_test, T_test):
        P, L = network.forward(X, T)
        test_loss += L

        if np.argmax(P) == T:
            test_accurate_count += 1

        test_count += 1

    # add mean test loss & accuracy to list
    test_losses.append(test_loss / test_count)
    test_accuracies.append(test_accurate_count / test_count)


x_axes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# losses
plt.plot(x_axes, training_losses, label = 'Training Loss')
plt.plot(x_axes, validation_losses, label = 'Validation Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(x_axes, test_losses, label = 'Test Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# accuracy
plt.plot(x_axes, validation_accuracies, label = 'Validation Accuracy')
plt.plot(x_axes, training_accuracies, label = 'Training Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.plot(x_axes, test_accuracies, label = 'Test Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()