import numpy
from network.Activations import Activations

class LinearAlgebra:

  @staticmethod
  def transpose(matrix: list) -> list:
    transpose = list()

    for i in range(len(matrix)):
      for j in range(len(matrix[i])):
        if len(transpose) < j + 1:
          transpose.append(list())

        transpose[j].append(matrix[i][j])

    return transpose

  @staticmethod
  def multiply(matrix1: list, matrix2: list) -> list:
    rows_matrix1 = len(matrix1)
    cols_matrix1 = len(matrix1[0])
    rows_matrix2 = len(matrix2)
    cols_matrix2 = len(matrix2[0])

    if cols_matrix1 != rows_matrix2:
      raise Exception("Cannot multiply the two matrices. Incorrect dimensions.")

    result = [[0 for row in range(cols_matrix2)] for col in range(rows_matrix1)]

    for i in range(rows_matrix1):
      for j in range(cols_matrix2):
        for k in range(cols_matrix1):
          result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result

  @staticmethod
  def add(matrix1: list, matrix2: list) -> list:
    rows_matrix1 = len(matrix1)
    cols_matrix1 = len(matrix1[0])
    rows_matrix2 = len(matrix2)
    cols_matrix2 = len(matrix2[0])

    if cols_matrix1 != cols_matrix2 and rows_matrix1 != rows_matrix2:
      raise Exception('Cannot add the two matrices. Incorrect dimensions.')

    result = [[0.0 for col in range(cols_matrix1)] for row in range(rows_matrix1)]

    for i in range(rows_matrix1):
      for j in range(cols_matrix1):
        result[i][j] = matrix1[i][j] + matrix2[i][j]

    return result

  @staticmethod
  def subtract(matrix1: list, matrix2: list) -> list:
    rows_matrix1 = len(matrix1)
    cols_matrix1 = len(matrix1[0])
    rows_matrix2 = len(matrix2)
    cols_matrix2 = len(matrix2[0])

    if cols_matrix1 != cols_matrix2 and rows_matrix1 != rows_matrix2:
      raise Exception('Cannot execute subtraction on the two matrices. Incorrect dimensions.')

    result = [[0.0 for col in range(cols_matrix1)] for row in range(rows_matrix1)]

    for i in range(rows_matrix1):
      for j in range(cols_matrix1):
        result[i][j] = matrix1[i][j] - matrix2[i][j]

    return result

  @staticmethod
  def zeros(dimensions: list) -> list:
    rows = dimensions[0]
    columns = dimensions[1]

    matrix = list()
    row = [0.0] * columns

    for i in range(rows):
      matrix.append(row)

    return matrix

  @staticmethod
  def ones(dimensions: list) -> list:
    rows = dimensions[0]
    columns = dimensions[1]

    matrix = list()
    row = [1.0] * columns

    for i in range(rows):
      matrix.append(row)

    return matrix

  @staticmethod
  def gaussian_random():
    numpy.random.seed(654)
    return 2 * numpy.random.random() - 1

  @staticmethod
  def random(dimensions: list) -> list:
    rows = dimensions[0]
    columns = dimensions[1]

    matrix = list()

    for i in range(rows):
      row = list()

      for j in range(columns):
        row.append(LinearAlgebra.gaussian_random())
      matrix.append(row)

    return matrix

  @staticmethod
  def apply(matrix: list, func) -> list:
    result = list()

    for i in range(len(matrix)):
      row = list()

      for j in range(len(matrix[i])):
        row.append(func(matrix[i][j]))

      result.append(row)

    return result

  @staticmethod
  def prettify(matrix: list) -> None:
    for i in range(len(matrix)):
      row = '* '
      for j in range(len(matrix[i])):
        row += str(matrix[i][j])

        if j < len(matrix[i]) - 1:
          row += ', '
      row += ' *'
      print(row)

  @staticmethod
  def vectorize(input_array: list) -> list:
    vector = list()

    for i in range(len(input_array)):
      row = list()
      row.append(input_array[i])
      vector.append(row)

    return vector

  @staticmethod
  def one_hot_encode(num_classes: int, value: int) -> list:
    one_hot_encoded = list()
    zero = list()
    zero.append(0)

    for i in range(num_classes):
      one_hot_encoded.append(zero)

    one = list()
    one.append(1)
    one_hot_encoded[value] = one

    return one_hot_encoded

  @staticmethod
  def softmax(matrix: list, func) -> list:
    result = list()

    for i in range(len(matrix)):
      row = list()

      for j in range(len(matrix[i])):
        row.append(func(matrix[i][j]))

      result.append(row)

    return result