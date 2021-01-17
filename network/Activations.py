import numpy

class Activations:

  @staticmethod
  def sigmoid(X):
    return 1 / (1 + numpy.exp(-X))

  @staticmethod
  def softmax(Y, target):
    return numpy.exp(Y[target]) / sum([numpy.exp(i) for i in Y])
