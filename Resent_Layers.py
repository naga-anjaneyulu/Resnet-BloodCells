
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

from tf_resnet_convblock import ConvLayer, BatchNormLayer, ConvBlock

class ReLULayer:
  def forward(self, X):
    return tf.nn.relu(X)

  def get_params(self):
    return []

class MaxPoolLayer:
  def __init__(self, dim):
    self.dim = dim

  def forward(self, X):
    return tf.nn.max_pool(
      X,
      ksize=[1, self.dim, self.dim, 1],
      strides=[1, 2, 2, 1],
      padding='VALID'
    )

  def get_params(self):
    return []

class PartialResNet:
  def __init__(self):
    self.layers = [
      # before conv block
      ConvLayer(d=7, mi=3, mo=64, stride=2, padding='SAME'),
      BatchNormLayer(64),
      ReLULayer(),
      MaxPoolLayer(dim=3),
      # conv block
      ConvBlock(mi=64, fm_sizes=[64, 64, 256], stride=1),
    ]
    self.input_ = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    self.output = self.forward(self.input_)

  def copyFromKerasLayers(self, layers):
    self.layers[0].copyFromKerasLayers(layers[1])
    self.layers[1].copyFromKerasLayers(layers[2])
    self.layers[4].copyFromKerasLayers(layers[5:])

  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)
    return X

  def predict(self, X):
    assert(self.session is not None)
    return self.session.run(
      self.output,
      feed_dict={self.input_: X}
    )

  def set_session(self, session):
    self.session = session
    self.layers[0].session = session
    self.layers[1].session = session
    self.layers[4].set_session(session)

  def get_params(self):
    params = []
    for layer in self.layers:
      params += layer.get_params()


if __name__ == '__main__':
  resnet = ResNet50(weights='imagenet')

  partial_model = Model(
    inputs=resnet.input,
    outputs=resnet.layers[16].output
  )
  print(partial_model.summary())
  my_partial_resnet = PartialResNet()
  X = np.random.random((1, 224, 224, 3))
  keras_output = partial_model.predict(X)
  init = tf.variables_initializer(my_partial_resnet.get_params())

  session = keras.backend.get_session()
  my_partial_resnet.set_session(session)
  session.run(init)

  first_output = my_partial_resnet.predict(X)
  print("first_output.shape:", first_output.shape)
  my_partial_resnet.copyFromKerasLayers(partial_model.layers)

  # compare the 2 models
  output = my_partial_resnet.predict(X)
  diff = np.abs(output - keras_output).sum()
  if diff < 1e-10:
    print("Everything's great!")
  else:
    print("diff = %s" % diff)

