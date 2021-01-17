import math
import random
import numpy
import pickle
import json
from copy import deepcopy
from datetime import datetime
import numpy as np

class MNISTNetwork:

  def __init__(self):
    # set initial values
    self.W = np.random.normal(size = (784,300))
    self.V = np.random.normal(size = (300,10))
    self.B = np.zeros(300)
    self.C = np.zeros(10)

    self.K = None
    self.H = None
    self.Y = None

  def forward(self, X, T):
    self.K = np.matmul(X, self.W) + self.B
    self.H = self.sigmoid(self.K)
    self.Y = np.matmul(self.H, self.V) + self.C

    S = self.softmax(self.Y)
    L = -np.log(S[T])

    return S, L

  def backward(self, X, T, S):
    dLdS = np.zeros(len(S))
    dLdS[T] = -1 / S[T]
    dLdY = -S * np.sum(dLdS * S) + dLdS * S
    dLdV = np.outer(self.H, dLdY)
    dLdC = dLdY
    V = self.V
    dLdH = np.dot(dLdY, V.T)
    dLdK = dLdH * self.H * (1 - self.H)
    dLdW = np.outer(X, dLdK)
    dLdB = dLdK

    return dLdV, dLdC, dLdW, dLdB

  def step(self, dLdV, dLdC, dLdW, dLdB, alpha):
    self.W -= alpha * dLdW
    self.B -= alpha * dLdB
    self.V -= alpha * dLdV
    self.C -= alpha * dLdC