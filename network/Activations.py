import random
import math

class Activations:

  @staticmethod
  def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

  @staticmethod
  def softmax(output: float, output_list: list) -> float:
    denominator = 0.0

    for i in range(len(output_list)):
      denominator += math.exp(output_list[i])

    return math.exp(output) / denominator