import random
import math

class Activations:

  @staticmethod
  def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

  @staticmethod
  def softmax(current: float, values: list) -> float:
    denominator = 0.0

    for i in range(len(values)):
      denominator += math.exp(values[i])

    return math.exp(current) / denominator
