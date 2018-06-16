"""Contains the definition for VX network. The VX network use a architecture
similar to that of inception_v1, with a few layer truncated/simplified."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from backbone import inception_utils

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


def inception_v1_base(inputs, final_endpoint='Mixed_5c', scope='InceptionV1'):
  """Defines the Inception V1 base architecture.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
      'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
      'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    scope: Optional variable_scope.

  Returns:
    A dictionary from components of the network to the corresponding activation.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values.
  """
  end_points = {}
  with tf.variable_scope(scope, 'InceptionV1', [inputs]):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_initializer=trunc_normal(0.01),
        activation_fn=tf.nn.leaky_relu):
      with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                          stride=1, padding='SAME'):
        end_point = 'Conv2d_1a_7x7'
        net = slim.conv2d(inputs, 64, [7, 7], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool_2a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'Conv2d_2b_1x1'
        net = slim.conv2d(net, 64, [1, 1], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'Conv2d_2c_3x3'
        net = slim.conv2d(net, 192, [3, 3], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool_3a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        #end_point = 'Mixed_3b'
        #with tf.variable_scope(end_point):
        #  with tf.variable_scope('Branch_0'):
        #    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
        #  with tf.variable_scope('Branch_1'):
        #    branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
        #    branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')
        #  with tf.variable_scope('Branch_2'):
        #    branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
        #    branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
        #  with tf.variable_scope('Branch_3'):
        #    branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
        #    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
        #  net = tf.concat(
        #      axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        #end_points[end_point] = net
        #if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool_4a_3x3'
        net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool_5a_2x2'
        net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4d'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4e'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4f'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        #end_point = 'MaxPool_5a_2x2'
        #net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
        #end_points[end_point] = net
        #if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          net = tf.concat(
              axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        end_points[end_point] = net
        if final_endpoint == end_point: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)

# You can change the size of `inputs'. We have modified the arch of the original
# inception v1 code, and now:
#     with `inputs' as [None, 224, 224, 3], shape of `net' will be [?, 7, 7, 1024].
#     with `inputs' as [None, 416, 416, 3], shape of `net' will be [?, 13, 13, 1024].
# 
# And we don't want global pooling at this level yet, so set global_pool=False.
def vx(inputs, is_training=True, reuse=None, scope='VX'):
  """Defines the VX architecture.

  The default image size used to train this network is 224x224.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    is_training: whether is training or not.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: a Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped-out input to the logits layer
      if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  with tf.variable_scope(scope, 'VX', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v1_base(inputs,
                                          final_endpoint='Mixed_4c',
                                          scope=scope)
      return net, end_points

vx_arg_scope = inception_utils.inception_arg_scope
