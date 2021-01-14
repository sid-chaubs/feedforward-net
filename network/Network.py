from network.Activations import Activations
from network.LinearAlgebra import LinearAlgebra as linalg
from network.Losses import Losses
from network.Definitions import DIM_W, DIM_V, DIM_B, DIM_C, DIM_K, DIM_H, DIM_Y

import math
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
    self.Y = linalg.zeros(DIM_Y)

  def reset(self):
    self.K = linalg.zeros(DIM_K)
    self.H = linalg.apply(self.K, Activations.sigmoid)
    self.Y = linalg.zeros(DIM_Y)

  def forward(self, X: list) -> [list, int]:
    P = [[0.], [0.]]

    # K = X * W + B
    self.K = linalg.add(
      linalg.multiply(linalg.transpose(self.W), X),
      self.B
    )

    # Calculate H = sigmoid(K)
    self.H = linalg.apply(self.K, Activations.sigmoid)

    # O = H * V + C
    self.Y = linalg.add(
      linalg.multiply(linalg.transpose(self.V), self.H),
      self.C
    )

    # softmax
    for j in range(len(self.Y)):
      P[j][0] = Activations.softmax_main(self.Y, j)

    return P

  def backwards(self, X: list, P: list, T: list) -> [list, list, list, list]:
    dLdP = [[0.], [0.]]
    dLdY = [[0.], [0.]]

    for i in range(len(dLdP)):
      dLdP[i][0] = -T[i][0] / P[i][0]

    dLdY[0][0] = (dLdP[0][0] - dLdP[1][0]) * P[0][0] * P[1][0]
    dLdY[1][0] = (dLdP[1][0] - dLdP[0][0]) * P[0][0] * P[1][0]
    dLdC = dLdY

    # Calculate the gradient of the loss (L) wrt to weights V of the hidden layer H, dL/dV = H * dL/dO
    dLdV = linalg.transpose(linalg.multiply(linalg.subtract(P, T), linalg.transpose(self.H)))
    dLdH = linalg.multiply(self.V, dLdY)

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

  def step(self, alpha, gradient: list, matrix: list) -> list:
    return linalg.subtract(
      matrix,
      linalg.apply(gradient, lambda val: val * alpha)
    )

  @staticmethod
  def SGD_update(W, b, V, c, W_d, b_d, V_d, c_d, alpha):
    for i in [0, 1]:
      for j in [0, 1, 2]:
        W[i][j] += -alpha * W_d[i][j]
  
    for i in range(len(b)):
      b[i][0] -= alpha * b_d[i]
  
    for i in [0, 1, 2]:
      for j in [0, 1]:
        V[i][j] -= alpha * V_d[i][j]
  
    for i in range(len(c)):
      c[i][0] -= alpha * c_d[i]

    return W, b, V, c

  @staticmethod
  def pass_backward(x, W, b, V, c, k, h, y, p, loss, t):
  
    # initialize derivatives
    p_d = [0., 0.]
    y_d = [0., 0.]
    V_d = [[0., 0.], [0., 0.], [0., 0.]]
    c_d = [0., 0.]
    h_d = [0., 0., 0.]
    k_d = [0., 0., 0.]
    W_d = [[0., 0., 0.], [0., 0., 0.]]
    b_d = [0., 0., 0.]
  
    for i in range(len(p_d)):
      p_d[i] = -t[i][0] / p[i]

    y_d[0] = (p_d[0] - p_d[1]) * p[0] * p[1]
    y_d[1] = (p_d[1] - p_d[0]) * p[0] * p[1]
  
    for j in range(len(y)):
      for i in range(len(h)):
        V_d[i][j] = (p[j] - t[j][0]) * h[i]
      c_d[j] = y_d[j]
  
    for i in range(len(h)):
      h_d[i] = y_d[0] * V[i][0] + y_d[1] * V[i][1]
  
    for i in range(len(k)):
      k_d[i] = h_d[i] * h[i] * (1 - h[i])
  
    for j in range(len(k)):
      for i in range(len(x)):
        W_d[i][j] = k_d[j] * x[i][0]
      b_d[i] = k_d[i]
  
    return W_d, b_d, V_d, c_d

  @staticmethod
  def pass_forward(x, W, b, V, c, t):
    k = [0., 0., 0.]
    h = [0., 0., 0.]
    y = [0., 0.]
    p = [0., 0.]

    for j in range(len(k)):
      for i in range(len(x)):
        k[j] += W[i][j] * x[i][0]
      k[j] += b[j][0]
      h[j] = Activations.sigmoid(k[j])

    for j in range(len(y)):
      for i in range(len(h)):
        y[j] += V[i][j] * h[i]
      y[j] += c[j][0]

    # softmax
    for j in range(len(y)):
      p[j] = Activations.softmax(y, j)

    loss = -(t[0][0] * math.log(p[0]) + t[1][0] * math.log(p[1]))

    return k, h, y, p, loss

  def optimize(self, alpha: float, epochs: int, dataset: list) -> list:
    inputs = dataset[0]
    targets = dataset[1]
    training_size = len(dataset[0])
    historical_losses = list()
    num_classes = 2
    historical_losses_2 = list()

    for current_epoch in range(epochs):
      total_loss = 0
      total_loss_2 = 0

      # calculate the loss
      for i in range(training_size):

        # init the input variable
        X = linalg.vectorize(inputs[i])
        T = linalg.one_hot_encode(num_classes, targets[i])
        P = self.forward(X)
        L = -(T[0][0] * math.log(P[0][0]) + T[1][0] * math.log(P[1][0]))

        k, h, y, p, loss = self.pass_forward(X, self.W, self.B, self.V, self.C, T)
        total_loss += L
        total_loss_2 += loss

        # calculate the local gradients
        dLdW, dLdB, dLdV, dLdC = self.backwards(X, P, T)
        W_d, b_d, V_d, c_d = self.pass_backward(X,self.W,self.B,self.V,self.C,k,h,y,p,loss,T)
        W,b,V,c = self.SGD_update(deepcopy(self.W), deepcopy(self.B), deepcopy(self.V), deepcopy(self.C), W_d, b_d, V_d, c_d, alpha)

        # take a step down the gradients
        self.W = self.step(alpha, dLdW, self.W)
        self.B = self.step(alpha, dLdB, self.B)
        self.V = self.step(alpha, dLdV, self.V)
        self.C = self.step(alpha, dLdC, self.C)

      historical_losses.append(total_loss / training_size)
      historical_losses_2.append(total_loss_2 / training_size)

      if current_epoch % 10 == 0:
        print(current_epoch, ' epoch.')

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
