import random
import math

class Activations:

  @staticmethod
  def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

  @staticmethod
  def softmax(x: list, index: int) -> float:
    denominator = sum([math.exp(i) for i in x])
    return math.exp(x[index]) / denominator
  
  @staticmethod
  def softmax_main(x: list, index: int) -> float:
    denominator = sum([math.exp(i[0]) for i in x])
    return math.exp(x[index][0]) / denominator