from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import captcha_input
import config

IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT
CLASSES_NUM = config.CLASSES_NUM
CHARS_NUM = config.CHARS_NUM

def inputs(train, batch_size):
    return captcha_input.inputs(train, batch_size=batch_size)


def _conv2d(value, weight):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(value, weight, strides=[1, 1, 1, 1], padding='SAME')


def _max_pool_2x2(value, name):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(value, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)


def _weight_variable(name, shape):
  """weight_variable generates a weight variable of a given shape."""
  with tf.device('/cpu:0'):
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    var = tf.get_variable(name,shape,initializer=initializer, dtype=tf.float32)
  return var


def _bias_variable(name, shape):
  """bias_variable generates a bias variable of a given shape."""
  with tf.device('/cpu:0'):
    initializer = tf.constant_initializer(0.1)
    var = tf.get_variable(name, shape, initializer=initializer,dtype=tf.float32)
  return var
  
def inference(images, keep_prob):
  images = tf.reshape(images, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
  
  with tf.variable_scope('conv1') as scope:
    kernel = _weight_variable('weights', shape=[3,3,1,64])
    biases = _bias_variable('biases',[64])
    pre_activation = tf.nn.bias_add(_conv2d(images, kernel),biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    
  pool1 = _max_pool_2x2(conv1, name='pool1')
  
  with tf.variable_scope('conv2') as scope:
    kernel = _weight_variable('weights', shape=[3,3,64,64])
    biases = _bias_variable('biases',[64])
    pre_activation = tf.nn.bias_add(_conv2d(pool1, kernel),biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    
  pool2 = _max_pool_2x2(conv2, name='pool2')
  
  with tf.variable_scope('conv3') as scope:
    kernel = _weight_variable('weights', shape=[3,3,64,64])
    biases = _bias_variable('biases',[64])
    pre_activation = tf.nn.bias_add(_conv2d(pool2, kernel),biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)
    
  pool3 = _max_pool_2x2(conv3, name='pool3')
  
  with tf.variable_scope('conv4') as scope:
    kernel = _weight_variable('weights', shape=[3,3,64,64])
    biases = _bias_variable('biases',[64])
    pre_activation = tf.nn.bias_add(_conv2d(pool3, kernel),biases)
    conv4 = tf.nn.relu(pre_activation, name=scope.name)
    
  pool4 = _max_pool_2x2(conv4, name='pool4')
  
  with tf.variable_scope('local1') as scope:
    batch_size = images.get_shape()[0].value
    reshape = tf.reshape(pool4, [batch_size,-1])
    dim = reshape.get_shape()[1].value
    weights = _weight_variable('weights', shape=[dim,1024])
    biases = _bias_variable('biases',[1024])
    local1 = tf.nn.relu(tf.matmul(reshape,weights) + biases, name=scope.name)

  local1_drop = tf.nn.dropout(local1, keep_prob)

  with tf.variable_scope('softmax_linear') as scope:
    weights = _weight_variable('weights',shape=[1024,CHARS_NUM*CLASSES_NUM])
    biases = _bias_variable('biases',[CHARS_NUM*CLASSES_NUM])
    softmax_linear = tf.add(tf.matmul(local1_drop,weights), biases, name=scope.name)

  return tf.reshape(softmax_linear, [-1, CHARS_NUM, CLASSES_NUM])


def loss(logits, labels):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                  labels=labels, logits=logits, name='corss_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def training(loss):
  optimizer = tf.train.AdamOptimizer(1e-4)
  train_op = optimizer.minimize(loss)
  return train_op


def evaluation(logits, labels):
  correct_prediction = tf.equal(tf.argmax(logits,2), tf.argmax(labels,2))
  correct_batch = tf.reduce_mean(tf.cast(correct_prediction, tf.int32), 1)
  return tf.reduce_sum(tf.cast(correct_batch, tf.float32))


def output(logits):
  return tf.argmax(logits, 2)

