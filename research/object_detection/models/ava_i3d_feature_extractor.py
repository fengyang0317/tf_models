# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Resnet V1 Faster R-CNN implementation.

See "Deep Residual Learning for Image Recognition" by He et al., 2015.
https://arxiv.org/abs/1512.03385

Note: this implementation assumes that the classification checkpoint used
to finetune this model is trained using the same configuration as that of
the MSRA provided checkpoints
(see https://github.com/KaimingHe/deep-residual-networks), e.g., with
same preprocessing, batch norm scaling, etc.
"""
import tensorflow as tf

from object_detection.meta_architectures import ava_meta_arch
from object_detection.utils.shape_utils import combined_static_and_dynamic_shape
from nets import resnet_utils
from nets import resnet_v1
import i3d

slim = tf.contrib.slim


class AvaI3dFeatureExtractor(ava_meta_arch.AvaFeatureExtractor):
  """Faster R-CNN Resnet V1 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               architecture='RGB',
               weight_decay=0.0):
    """Constructor.

    Args:
      architecture: Architecture name of the Resnet V1 model.
      resnet_model: Definition of the Resnet V1 model.
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride != 8 and first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 8 or 16.')
    self._architecture = architecture
    super(AvaI3dFeatureExtractor, self).__init__(
      is_training, first_stage_features_stride, batch_norm_trainable,
      reuse_weights, weight_decay)

  def preprocess(self, resized_inputs):
    """Faster R-CNN Resnet V1 preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.

    """
    pass

  def _extract_proposal_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float32 tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      rpn_feature_map: A tensor with shape [batch, height, width, depth]
      activations: A dictionary mapping feature extractor tensor names to
        tensors

    Raises:
      InvalidArgumentError: If the spatial size of `preprocessed_inputs`
        (height or width) is less than 33.
      ValueError: If the created network is missing the required activation.
    """
    if len(preprocessed_inputs.get_shape().as_list()) != 5:
      raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                       'tensor of shape %s' % preprocessed_inputs.get_shape())
    shape_assert = tf.Assert(
      tf.logical_and(
        tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33),
        tf.greater_equal(tf.shape(preprocessed_inputs)[3], 33)),
      ['image size must at least be 33 in both height and width.'])

    with tf.control_dependencies([shape_assert]):
      # Disables batchnorm for fine-tuning with smaller batch sizes.
      # TODO(chensun): Figure out if it is needed when image
      # batch size is bigger.
      with tf.variable_scope(
          self._architecture, reuse=self._reuse_weights) as var_scope:
        rgb_model = i3d.InceptionI3d(2, spatial_squeeze=True,
                                     final_endpoint='Mixed_4c')
        _, activations = rgb_model(preprocessed_inputs,
                                   is_training=self._train_batch_norm,
                                   dropout_keep_prob=1.0)

    handle = 'Mixed_4c'
    feat = activations[handle]
    feat = tf.transpose(feat, [0, 2, 3, 1, 4])
    shape = combined_static_and_dynamic_shape(feat)
    feat = tf.reshape(feat, shape[:3] + [shape[3] * shape[4]])
    return feat, activations

  def _extract_query_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float32 tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      rpn_feature_map: A tensor with shape [batch, height, width, depth]
      activations: A dictionary mapping feature extractor tensor names to
        tensors

    Raises:
      InvalidArgumentError: If the spatial size of `preprocessed_inputs`
        (height or width) is less than 33.
      ValueError: If the created network is missing the required activation.
    """

    # if len(preprocessed_inputs.get_shape().as_list()) != 5:
    #   raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
    #                    'tensor of shape %s' % preprocessed_inputs.get_shape())
    # shape_assert = tf.Assert(
    #   tf.logical_and(
    #     tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
    #     tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
    #   ['image size must at least be 33 in both height and width.'])
    shape_assert = tf.no_op()

    def reshape(x):
      shape = combined_static_and_dynamic_shape(x)
      x = tf.reshape(x, [-1, 1, 8] + shape[1:])
      return x

    preprocessed_inputs = [reshape(i) for i in preprocessed_inputs]

    with tf.control_dependencies([shape_assert]):
      # Disables batchnorm for fine-tuning with smaller batch sizes.
      # TODO(chensun): Figure out if it is needed when image
      # batch size is bigger.
      with tf.variable_scope(
          self._architecture, reuse=True) as var_scope:
        rgb_model = i3d.InceptionI3d(2, spatial_squeeze=True,
                                     final_endpoint='Mixed_4c')

        def body(x):
          _, activations = rgb_model(x,
                                     is_training=self._train_batch_norm,
                                     dropout_keep_prob=1.0)
          handle = 'Mixed_4c'
          feat = activations[handle]
          feat = tf.transpose(feat, [0, 2, 3, 1, 4])
          shape = combined_static_and_dynamic_shape(feat)
          feat = tf.reshape(feat, shape[:3] + [shape[3] * shape[4]])
          return feat[0]

        ret = [
          tf.map_fn(body, i, tf.float32, parallel_iterations=1, back_prop=False)
          for i in preprocessed_inputs]
    return ret

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name (unused).

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    with tf.variable_scope(self._architecture, reuse=self._reuse_weights):
      with slim.arg_scope(
          resnet_utils.resnet_arg_scope(
            batch_norm_epsilon=1e-5,
            batch_norm_scale=True,
            weight_decay=self._weight_decay)):
        with slim.arg_scope([slim.batch_norm],
                            is_training=self._train_batch_norm):
          blocks = [
            resnet_utils.Block('block4', resnet_v1.bottleneck, [{
              'depth': 2048,
              'depth_bottleneck': 512,
              'stride': 1
            }] * 3)
          ]
          proposal_classifier_features = resnet_utils.stack_blocks_dense(
            proposal_feature_maps, blocks)
    return proposal_classifier_features
