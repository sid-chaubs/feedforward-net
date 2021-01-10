from helpers.data import load_synth
from network.Network import Network
import matplotlib.pyplot as plt
from network.LinearAlgebra import LinearAlgebra

import random
import math
import sys
import traceback

# gradient descent
network = Network()

print('Load existing model? Y / n')

load = input()

if load == 'Y':
  print('Enter file name:')
  filename = input()
  network.load(filename)
  print('Model loaded successfully.')
else:
  print('Training a new model.')

  # load the data
  data = load_synth()
  training_data = data[0]
  validation_data = data[1]
  learning_rate = 0.01
  epochs = 30 # a full pass through the training data

  loss_history = network.gradient_descent(learning_rate, epochs, training_data)
  count_accurate = 0
  test_set = validation_data

  for i in range(len(test_set[0])):
    prediction = network.predict(test_set[0][i])
    to_array = [item for sublist in prediction for item in sublist]
    predicted_class = to_array.index(max(to_array))
    expected_class = int(test_set[1][i])

    if predicted_class == expected_class:
      count_accurate += 1

  print('Save current model? Y / n')

  save = input()
  if save == 'Y':
    print('Enter file name:')
    filename = input()
    network.save(filename)

  print('Accuracy: ', count_accurate, '/', len(training_data[0]))