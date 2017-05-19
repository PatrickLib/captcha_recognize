from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf

import config

RECORD_DIR = config.RECORD_DIR
TRAIN_FILE = config.TRAIN_FILE
VALID_FILE = config.VALID_FILE

IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT
CLASSES_NUM = config.CLASSES_NUM
CHARS_NUM = config.CHARS_NUM

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label_raw': tf.FixedLenFeature([], tf.string),
      })
  image = tf.decode_raw(features['image_raw'], tf.int16)
  image.set_shape([IMAGE_HEIGHT * IMAGE_WIDTH])
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  reshape_image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 1])
  label = tf.decode_raw(features['label_raw'], tf.uint8)
  label.set_shape([CHARS_NUM * CLASSES_NUM])
  reshape_label = tf.reshape(label, [CHARS_NUM, CLASSES_NUM])
  return tf.cast(reshape_image, tf.float32), tf.cast(reshape_label, tf.float32)


def inputs(train, batch_size):
  filename = os.path.join(RECORD_DIR,
                          TRAIN_FILE if train else VALID_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename])
    image, label = read_and_decode(filename_queue)
    if train:
        images, sparse_labels = tf.train.shuffle_batch([image, label],
                                                       batch_size=batch_size,
                                                       num_threads=6,
                                                       capacity=2000 + 3 * batch_size,
                                                       min_after_dequeue=2000)
    else:
        images, sparse_labels = tf.train.batch([image, label],
                                               batch_size=batch_size,
                                               num_threads=6,
                                               capacity=2000 + 3 * batch_size)

    return images, sparse_labels
