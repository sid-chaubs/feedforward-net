from network.Activations import Activations
import numpy
import warnings

class GenericNetwork:

  def __init__(self):
    # weights and biases
    self.W = numpy.random.normal(scale=1.0,size=(2,3)).tolist()
    self.V = numpy.random.normal(scale = 1.0, size = (3, 2)).tolist()
    self.B = [0., 0., 0.]
    self.C = [0., 0.]

    # hidden layers
    self.K = [0., 0., 0.]
    self.H = [0., 0., 0.]
    self.Y = [0., 0.]
  

  def forward(self, X, T) -> [list, int]:
    self.K = [0., 0., 0.]
    self.H = [0., 0., 0.]
    self.Y = [0., 0.]
    P = [0., 0.]

    for j in range(len(self.K)):
      for i in range(len(X)):
        self.K[j] += self.W[i][j] * X[i]

      self.K[j] += self.B[j]
      self.H[j] = Activations.sigmoid(self.K[j])
  
    for j in range(len(self.Y)):
      for i in range(len(self.H)):
        self.Y[j] += self.V[i][j] * self.H[i]
      self.Y[j] += self.C[j]


    for i in range(len(self.Y)):
      P[i] = Activations.softmax(self.Y, i)

    L = -(T[0] * numpy.log(P[0]) + T[1] * numpy.log(P[1]))

    return P, L

  def backward(self, X: list, P: list, T: list) -> [int, int, int, int]:
    dLdP = [0.,0.]
    dLdY = [0.,0.]
    dLdV = [[0.,0.],[0.,0.],[0.,0.]]
    dLdC = [0.,0.]
    dLdH = [0.,0.,0.]
    dLdK = [0.,0.,0.]
    dLdW = [[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
    dLdB = [0., 0., 0.]

    for i in range(len(dLdP)):
        dLdP[i] = -T[i] / P[i]

    dLdY[0]= (dLdP[0] - dLdP[1]) * P[0] * P[1]
    dLdY[1]= (dLdP[1] - dLdP[0]) * P[0] * P[1]

    for j in range(len(self.Y)):
        for i in range(len(self.H)):
            dLdV[i][j] = (P[j] - T[j]) * self.H[i]

        dLdC[j] = dLdY[j]

    for i in range(len(self.H)):
        dLdH[i] = dLdY[0] * self.V[i][0] + dLdY[1] * self.V[i][1]

    for i in range(len(self.K)):
        dLdK[i] = dLdH[i] * self.H[i] * (1 - self.H[i])

    for j in range(len(self.K)):
        for i in range(len(X)):
            dLdW[i][j] = dLdK[j] * X[i]

        dLdB[i] = dLdK[i]

    return dLdW, dLdB, dLdV, dLdC

  def step(self, dLdW, dLdB, dLdV, dLdC, alpha) -> None:
    for i in range(len(self.W)):
      for j in range(len(self.W[i])):
        self.W[i][j] += -alpha * dLdW[i][j]
  
    for i in range(len(self.B)):
      self.B[i] -= alpha * dLdB[i]
  
    for i in range(len(self.V)):
      for j in range(len(self.V[i])):
        self.V[i][j] -= alpha * dLdV[i][j]
  
    for i in range(len(self.C)):
      self.C[i] -= alpha * dLdC[i]
