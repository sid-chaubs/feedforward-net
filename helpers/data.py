import numpy as np
from urllib import request
import gzip
import pickle
import os

DATA_FILENAMES = [
  ['training_images', './data/train-images-idx3-ubyte.gz'],
  ['test_images', './data/t10k-images-idx3-ubyte.gz'],
  ['training_labels', './data/train-labels-idx1-ubyte.gz'],
  ['test_labels', './data/t10k-labels-idx1-ubyte.gz']
]

MNIST_IMAGE_WIDTH = 28
MNIST_IMAGE_HEIGHT = 28
MNIST_IMAGE_MAX_PIXEL_VALUE = 255

def load_synth(num_train = 60_000, num_val = 10_000):
  """
  Load some very basic synthetic data that should be easy to classify. Two features, so that we can plot the
  decision boundary (which is an ellipse in the feature space).
  :param num_train: Number of training instances
  :param num_val: Number of test/validation instances
  :param num_features: Number of features per instance
  :return: Two tuples (xtrain, ytrain), (xval, yval) the training data is a floating point numpy array:
  """

  THRESHOLD = 0.6
  quad = np.asarray([[1, 0.5], [1, .2]])

  total = num_train + num_val

  x = np.random.randn(total, 2)

  # compute the quadratic form
  q = np.einsum('bf, fk, bk -> b', x, quad, x)
  y = (q > THRESHOLD).astype(np.int)

  return (x[:num_train, :], y[:num_train]), (x[num_train:, :], y[num_train:]), 2

def load_mnist(final = False, flatten = True):
  """
    Load the MNIST data
    :param final: If true, return the canonical test/train split. If false, split some validation data from the training
       data and keep the test data hidden.
    :param flatten:
    :return:
    """

  if not os.path.isfile('./data/mnist.pkl'):
    init_mnist()

  xtrain, ytrain, xtest, ytest = load_mnist()
  xtl, xsl = xtrain.shape[0], xtest.shape[0]

  if flatten:
    xtrain = xtrain.reshape(xtl, -1)
    xtest = xtest.reshape(xsl, -1)

  if not final:  # return the flattened images
    return (xtrain[:-5000], ytrain[:-5000]), (xtrain[-5000:], ytrain[-5000:]), 10

  return (xtrain, ytrain), (xtest, ytest), 10

def download_mnist():
  base_url = 'http://yann.lecun.com/exdb/mnist/'

  for name in DATA_FILENAMES:
    print('Downloading ' + name[1] + '...')
    request.urlretrieve(base_url + name[1], name[1])

  print('Download complete.')

def save_mnist():
  mnist = {}
  for name in DATA_FILENAMES[:2]:
    with gzip.open(name[1], 'rb') as f:
      mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset = 16).reshape(-1, 28 * 28)
  for name in DATA_FILENAMES[-2:]:
    with gzip.open(name[1], 'rb') as f:
      mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset = 8)
  with open('./data/mnist.pkl', 'wb') as f:
    pickle.dump(mnist, f)
  print('Save complete.')

def init_mnist():
  download_mnist()
  save_mnist()

def normalize(data):
  return data / MNIST_IMAGE_MAX_PIXEL_VALUE

def load_mnist():
  with open('mnist.pkl', 'rb') as f:
    mnist = pickle.load(f)

  data = dict()

  data['training_images'] = normalize(mnist['training_images'])
  data['training_labels'] = mnist['training_labels']

  data['test_images'] = normalize(mnist['test_images'])
  data['test_labels'] = mnist['test_labels']

  return data