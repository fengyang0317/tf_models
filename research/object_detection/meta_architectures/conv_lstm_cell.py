from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl


class ConvLSTMCell(rnn_cell_impl.RNNCell):
  """Convolutional LSTM recurrent network cell.

  https://arxiv.org/pdf/1506.04214v1.pdf
  """

  def __init__(self,
               conv_ndims,
               input_shape,
               output_channels,
               kernel_shape,
               use_bias=True,
               skip_connection=True,
               forget_bias=1.0,
               initializers=None,
               name="conv_lstm_cell"):
    """Construct ConvLSTMCell.

    Args:
      conv_ndims: Convolution dimensionality (1, 2 or 3).
      input_shape: Shape of the input as int tuple, excluding the batch size.
      output_channels: int, number of output channels of the conv LSTM.
      kernel_shape: Shape of kernel as in tuple (of size 1,2 or 3).
      use_bias: (bool) Use bias in convolutions.
      skip_connection: If set to `True`, concatenate the input to the
        output of the conv LSTM. Default: `False`.
      forget_bias: Forget bias.
      initializers: Unused.
      name: Name of the module.

    Raises:
      ValueError: If `skip_connection` is `True` and stride is different from 1
        or if `input_shape` is incompatible with `conv_ndims`.
    """
    super(ConvLSTMCell, self).__init__(name=name)

    if conv_ndims != len(input_shape) - 1:
      raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(
        input_shape, conv_ndims))

    self._conv_ndims = conv_ndims
    self._input_shape = input_shape
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._use_bias = use_bias
    self._forget_bias = forget_bias
    self._skip_connection = skip_connection

    self._total_output_channels = output_channels

    state_size = tensor_shape.TensorShape(
      self._input_shape[:-1] + [self._output_channels])
    self._state_size = rnn_cell_impl.LSTMStateTuple(state_size, state_size)
    self._output_size = tensor_shape.TensorShape(
      self._input_shape[:-1] + [self._total_output_channels])

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs, state, scope=None):
    cell, hidden = state
    new_inputs = tf.concat([inputs, hidden], axis=-1)
    off = slim.separable_conv2d(new_inputs, 2, [3, 3], depth_multiplier=1,
                                stride=5, padding='SAME', activation_fn=None)
    off = off[:, 1:, 1:]
    batch_size = inputs._shape_as_list()[0]
    off = tf.reshape(off, [-1, 9, 2])
    r = np.arange(3,dtype=np.float32) * 5 + 5;
    cx = np.tile(r, [3]);
    r.shape = 3, 1
    cy = np.tile(r, [1, 3])
    cy.shape = -1
    control_points = np.stack([cy, cx], axis=1)
    control_points = np.expand_dims(control_points, axis=0)
    control_points = tf.tile(control_points, [batch_size, 1, 1])
    hidden, _ = tf.contrib.image.sparse_image_warp(
      hidden,
      control_points,
      control_points + off,
      num_boundary_points=4)

    if 'subset' not in tf.flags.FLAGS:
      grid = np.ones([201, 201], dtype=np.float32)
      for i in range(0, 201, 50):
        grid[i, :] = 0
        grid[:, i] = 0
      grid = np.expand_dims(grid, axis=0)
      grid = np.tile(grid, [batch_size, 1, 1])
      grid, _ = tf.contrib.image.sparse_image_warp(
        grid[:, :, :, np.newaxis],
        control_points * 10,
        (control_points + off) * 10,
        num_boundary_points=4)
      tf.summary.image('grid', grid)

    new_inputs = tf.concat([inputs, hidden], axis=-1)
    new_hidden = slim.separable_conv2d(new_inputs, 128,
                                       self._kernel_shape, depth_multiplier=1,
                                       activation_fn=None)
    new_hidden = slim.conv2d(new_hidden, 4 * self._output_channels, [1, 1],
                             activation_fn=None)
    gates = array_ops.split(
      value=new_hidden, num_or_size_splits=4, axis=self._conv_ndims + 1)

    input_gate, new_input, forget_gate, output_gate = gates
    input_gate = tf.contrib.layers.layer_norm(input_gate, scope='input')
    new_input = tf.contrib.layers.layer_norm(new_input, scope='transform')
    forget_gate = tf.contrib.layers.layer_norm(forget_gate, scope='forget')
    output_gate = tf.contrib.layers.layer_norm(output_gate, scope='output')
    new_cell = math_ops.sigmoid(forget_gate + self._forget_bias) * cell
    new_cell += math_ops.sigmoid(input_gate) * math_ops.tanh(new_input)
    new_cell = tf.contrib.layers.layer_norm(new_cell, scope='state')
    output = math_ops.tanh(new_cell) * math_ops.sigmoid(output_gate)

    if self._skip_connection:
      output = output + inputs
    new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
    return output, new_state
