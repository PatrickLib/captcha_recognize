from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import config

IMAGE_HEIGHT = config.IMAGE_HEIGHT
IMAGE_WIDTH = config.IMAGE_WIDTH
CHAR_SETS = config.CHAR_SETS
CLASSES_NUM = config.CLASSES_NUM
CHARS_NUM = config.CHARS_NUM

RECORD_DIR = config.RECORD_DIR
TRAIN_FILE = config.TRAIN_FILE
VALID_FILE = config.VALID_FILE

FLAGS = None

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  
def label_to_one_hot(label):
  one_hot_label = np.zeros([CHARS_NUM, CLASSES_NUM])
  offset = []
  index = []
  for i, c in enumerate(label):
    offset.append(i)
    index.append(CHAR_SETS.index(c))
  one_hot_index = [offset, index]
  one_hot_label[one_hot_index] = 1.0
  return one_hot_label.astype(np.uint8)


def conver_to_tfrecords(data_set, name):
  """Converts a dataset to tfrecords."""
  if not os.path.exists(RECORD_DIR):
      os.makedirs(RECORD_DIR)
  filename = os.path.join(RECORD_DIR, name)
  print('>> Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  num_examples = len(data_set)
  for index in range(num_examples):
    image = data_set[index][0]
    height = image.shape[0]
    width = image.shape[1]
    image_raw = image.tostring()
    label = data_set[index][1]
    label_raw = label_to_one_hot(label).tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'label_raw': _bytes_feature(label_raw),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()
  print('>> Writing Done!')
  

def create_data_list(image_dir):
  if not gfile.Exists(image_dir):
    print("Image director '" + image_dir + "' not found.")
    return None
  extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
  print("Looking for images in '" + image_dir + "'")
  file_list = []
  for extension in extensions:
    file_glob = os.path.join(image_dir, '*.' + extension)
    file_list.extend(gfile.Glob(file_glob))
  if not file_list:
    print("No files found in '" + image_dir + "'")
    return None
  images = []
  labels = []
  for file_name in file_list:
    image = Image.open(file_name)
    image_gray = image.convert('L')
    image_resize = image_gray.resize(size=(IMAGE_WIDTH,IMAGE_HEIGHT))
    input_img = np.array(image_resize, dtype='int16')
    image.close()
    label_name = os.path.basename(file_name).split('_')[0]
    images.append(input_img)
    labels.append(label_name)
  return zip(images, labels)


def main(_):
  training_data = create_data_list(FLAGS.train_dir)
  conver_to_tfrecords(training_data, TRAIN_FILE)
    
  validation_data = create_data_list(FLAGS.valid_dir)
  conver_to_tfrecords(validation_data, VALID_FILE)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_dir',
      type=str,
      default='./data/train_data',
      help='Directory training to get captcha data files and write the converted result.'
  )
  parser.add_argument(
      '--valid_dir',
      type=str,
      default='./data/valid_data',
      help='Directory validation to get captcha data files and write the converted result.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
