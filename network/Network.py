from network.Activations import Activations
from network.LinearAlgebra import LinearAlgebra as linalg
from network.Losses import Losses
from network.Definitions import DIM_W, DIM_V, DIM_B, DIM_C, DIM_K, DIM_H, DIM_O

import random
import numpy
import pickle
import json
from copy import deepcopy
from datetime import datetime

class Network:

  def __init__(self):
    # weights and biases
    self.W = linalg.random(DIM_W)
    self.V = linalg.random(DIM_V)
    self.B = linalg.zeros(DIM_B)
    self.C = linalg.zeros(DIM_C)

    # hidden layers
    self.K = linalg.zeros(DIM_K)
    self.H = linalg.apply(self.K, Activations.sigmoid)
    self.O = linalg.zeros(DIM_O)

  def reset(self):
    self.K = linalg.zeros(DIM_K)
    self.H = linalg.apply(self.K, Activations.sigmoid)
    self.O = linalg.zeros(DIM_O)

  def forward(self, X) -> list:
    self.reset()

    # K = X * W + B
    self.K = linalg.add(
      linalg.multiply(linalg.transpose(self.W), X),
      self.B
    )

    # Calculate H = sigmoid(K)
    self.H = linalg.apply(self.K, Activations.sigmoid)

    # O = H * V + C
    self.O = linalg.add(
      linalg.multiply(linalg.transpose(self.V), self.H),
      self.C
    )

    # apply softmax
    return self.output(self.O)

  def output(self):
    Y = list()
    linear_outputs = [item for sublist in self.O for item in sublist]

    for i in range(len(linear_outputs)):
      probability = list()
      probability.append(Activations.softmax(deepcopy(linear_outputs[i]), deepcopy(linear_outputs)))
      Y.append(probability)

    return  Y

  def calculate_gradients(self, X: list, Y: list, T: list) -> [list, list, list, list]:

    # Calculate the gradient of the loss (L) wrt to the linear outputs (O), dL/dO = Y - T
    dLdO = linalg.subtract(Y, T)

    # Calculate the gradient of the loss (L) wrt to the bias C
    dLdC = dLdO

    # Calculate the gradient of the loss (L) wrt to weights V of the hidden layer H, dL/dV = H * dL/dO
    dLdV = linalg.multiply(self.H, linalg.transpose(dLdO))

    # Calculate the gradient of the loss (L) wrt to the hidden layer H, dL/dH = V * dL/dO
    dLdH = linalg.multiply(self.V, dLdO)

    # Calculate the gradient of the loss (L) wrt to K, dL/dK = dL/dH * H * (1 - H)
    dLdK = linalg.multiply(
      dLdH,
      # H * (1 - H)
      linalg.multiply(
        linalg.transpose(
          # 1 - H
          linalg.subtract(linalg.ones(DIM_H), self.H)
        ),
        self.H
      )
    )

    # Calculate the gradient of the loss (L) wrt to the bias B, dLdB = dLdK
    dLdB = dLdK

    # Calculate the gradient of the loss (L) wrt to weights W, dL/dW = dL/dK * X
    dLdW = linalg.multiply(X, linalg.transpose(dLdK))

    return dLdW, dLdB, dLdV, dLdC

  def gradient_step(self, alpha, gradient: list, matrix: list) -> list:
    return linalg.subtract(
      matrix,
      linalg.apply(gradient, lambda val: val * alpha)
    )

  def gradient_descent(self, alpha: float, epochs: int, dataset: list) -> list:
    inputs = dataset[0]
    targets = dataset[1]
    training_size = len(dataset[0])
    historical_losses = list()
    num_classes = 2
    num_zeros = 0

    for current_epoch in range(epochs):
      L = 0

      # calculate the loss
      for i in range(training_size):

        # init the input variable
        X = linalg.vectorize(inputs[i])
        Y = self.forward(deepcopy(X))
        T = linalg.one_hot_encode(num_classes, targets[i])

        if int(targets[i]) == 0:
          num_zeros += 1

        # calculate the local gradients
        dLdW, dLdB, dLdV, dLdC = self.calculate_gradients(deepcopy(X), deepcopy(Y), deepcopy(T))

        # take a step down the gradients
        self.W = self.gradient_step(alpha, dLdW, deepcopy(self.W))
        self.B = self.gradient_step(alpha, dLdB, deepcopy(self.B))
        self.V = self.gradient_step(alpha, dLdV, deepcopy(self.V))
        self.C = self.gradient_step(alpha, dLdC, deepcopy(self.C))

        # get the loss once we take a step
        # get the output of our network
        L += Losses.cross_entropy(deepcopy(Y), deepcopy(T))

      mean_loss = L / len(dataset[0])
      historical_losses.append(mean_loss)
      print('Epoch... ', current_epoch, ', Loss: ', mean_loss)

    return historical_losses

  def predict(self, X: list) -> list:
    return self.forward(linalg.vectorize(X))

  def save(self, filename = datetime.now()) -> None:
    json_content = json.dumps(self.__dict__)

    f = open(f'network/models/{str(filename)}', 'w')
    f.write(json_content)
    f.close()

  def load(self, filename) -> None:
    with open(f'network/models/{filename}') as model:
      parameters = json.load(model)
      self.__dict__ = parameters