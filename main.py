from helpers.data import load_synth
from network.Network import Network
import matplotlib.pyplot as plt

import random
import math
import sys
import traceback

# load the data
data = load_synth()
training_data = data[0]
validation_data = data[1]

# epoch implies a full pass over the entire dataset
epochs = 5000
learning_rate = 0.0001

# gradient descent
network = Network()
loss_history = network.gradient_descent(learning_rate, epochs, training_data)
count_accurate = 0

for i in range(len(training_data[0])):
  prediction = network.predict(training_data[0][i])
  output = float(prediction.index(max(prediction)))
  expected = float(training_data[1][i])

  print(f'Predicted {output}, Expected {expected}')

  if output == expected:
    count_accurate += 1

print('Accuracy: ', count_accurate, '/', len(training_data[0]))

# plot the historical losses
plt.plot(loss_history)
plt.ylabel('Loss')
plt.show()