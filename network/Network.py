from network.Activations import Activations
from network.Losses import Losses
import random
import numpy as np
from copy import deepcopy

class Network:

  def __init__(self):
    # weights
    self.W = None
    self.B = None
    self.V = None
    self.C = None

    # hidden layers
    self.K = None
    self.H = None
    self.O = None

    # First set of weights mapping X to K
    self.initialize_weights()
    self.initialize_hidden_layers()

  def init_weight(self) -> float:
    return 2 * np.random.random() - 1

  def initialize_weights(self) -> None:
    np.random.seed(1)

    self.W = [
      [self.init_weight(), self.init_weight(), self.init_weight()],
      [self.init_weight(), self.init_weight(), self.init_weight()]
    ]

    # Second set of weights mapping H to the linear output being fed to the Softmax function
    self.V = [
      [self.init_weight(), self.init_weight()],
      [self.init_weight(), self.init_weight()],
      [self.init_weight(), self.init_weight()]
    ]

    # Bias for the first layer
    self.B = [0.0, 0.0, 0.0]

    # Second set of biases
    self.C = [0.0, 0.0]

  def initialize_hidden_layers(self) -> None:
    # Linear outputs for the first hidden layer
    self.K = [0.0, 0.0, 0.0]

    # Output of the sigmoid function applied to K, H = sigmoid(K)
    self.H = [Activations.sigmoid(self.K[0]), Activations.sigmoid(self.K[1]), Activations.sigmoid(self.K[2])]

    # Linear output fed to the Softmax function
    self.O = [0.0, 0.0]

  @staticmethod
  def multiply(self, X, Y):
    for i in range(len(X)):
      # iterate through columns of Y
      for j in range(len(Y[0])):
        # iterate through rows of Y
        for k in range(len(Y)):
          result[i][j] += X[i][k] * Y[k][j]

  def forward_propagate(self, X) -> list:
    # Calculate K = X * W + B
    self.K = [0.0, 0.0, 0.0]

    for i in range(len(self.W)):
      for j in range(len(self.W[i])):
        self.K[j] += X[i] * self.W[i][j]

    for i in range(len(self.B)):
      self.K[i] += self.B[i]

    # Calculate H = sigmoid(K)
    self.H = [0.0, 0.0, 0.0]

    for i in range(len(self.H)):
      self.H[i] = Activations.sigmoid(self.K[i])

    # Calculate O = H * V + C
    self.O = [0.0, 0.0]

    for i in range(len(self.V)):
      for j in range(len(self.V[i])):
        self.O[j] += self.H[i] * self.V[i][j]

    for i in range(len(self.O)):
      self.O[i] += self.C[i]

    return [ Activations.softmax(self.O[0], self.O), Activations.softmax(self.O[1], self.O) ]

  def calculate_gradients(self, X: list, Y: list, T: list) -> [list, list, list, list]:
    # Calculate the gradient of the loss (L) wrt to the linear outputs
    dLdO = [0.0, 0.0]

    for i in range(len(Y)):
      dLdO[i] = Y[i] - T[i]

    # Calculate the gradient of the loss (L) wrt to the bias C
    dLdC = dLdO

    # Calculate the gradient of the loss (L) wrt to weights V of the hidden layer H
    dLdV = [
      [0.0, 0.0],
      [0.0, 0.0],
      [0.0, 0.0]
    ]

    for i in range(len(self.V)):
      for j in range(len(self.V[i])):
        dLdV[i][j] = self.H[i] * dLdO[j]

    # Calculate the gradient of the loss (L) wrt to K
    dLdK = [0.0, 0.0, 0.0]

    for i in range(len(dLdK)):
      dLdK[i] = Activations.sigmoid(self.K[i]) * Activations.sigmoid(1.0 - self.K[i])

    # Calculate the gradient of the loss (L) wrt to weights W
    dLdW = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0]
    ]

    for i in range(len(dLdW)):
      for j in range(len(dLdW[i])):
        dLdW[i][j] = X[i] * dLdK[j]

    # Calculate the gradient of the loss (L) wrt to the bias B
    dLdB = dLdK

    return dLdW, dLdB, dLdV, dLdC

  def step(self, learning_rate, dLdW, dLdB, dLdV, dLdC) -> None:
    # Update C based on the learning rate and dL / dC
    for i in range(len(self.C)):
      self.C[i] = self.C[i] - learning_rate * dLdC[i]

    # Update V based on the learning rate and dL / dV
    for i in range(len(self.V)):
      for j in range(len(self.V[i])):
        self.V[i][j] = self.V[i][j] - learning_rate * dLdV[i][j]

    # Update B based on the learning rate and dL / dB
    for i in range(len(self.B)):
      self.B[i] = self.B[i] - learning_rate * dLdB[i]

    # Update W based on the learning rate and dL / dW
    for i in range(len(self.W)):
      for j in range(len(self.W[i])):
        self.W[i][j] = self.W[i][j] - learning_rate * dLdW[i][j]

  def gradient_descent(self, learning_rate: int, epochs: int, dataset: list) -> list:
    inputs = dataset[0]
    targets = dataset[1]
    historical_losses = list()
    data_size = float(len(dataset))

    for current_epoch in range(epochs):
      # mean loss
      L = 0

      # calculate the loss
      for i in range(len(dataset)):
        X = inputs[i]
        T = [0.0, 0.0]
        T[targets[i]] = 1.0

        # get the output of our network
        Y = self.forward_propagate(deepcopy(X))

        L += Losses.cross_entropy(deepcopy(Y), deepcopy(T))

        # calculate gradients wrt loss
        dLdW, dLdB, dLdV, dLdC = self.calculate_gradients(deepcopy(X), deepcopy(Y), deepcopy(T))

      # take a step down the gradient
      self.step(learning_rate, dLdW, dLdB, dLdV, dLdC)
      historical_losses.append(L / len(dataset))

    return historical_losses

  def predict(self, X: list) -> int:
    return self.forward_propagate(X)
