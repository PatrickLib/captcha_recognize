from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os.path
from datetime import datetime
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile
import captcha_model as captcha

import config

IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT

CHAR_SETS = config.CHAR_SETS
CLASSES_NUM = config.CLASSES_NUM
CHARS_NUM = config.CHARS_NUM

FLAGS = None

def one_hot_to_texts(recog_result):
  texts = []
  for i in xrange(recog_result.shape[0]):
    index = recog_result[i]
    texts.append(''.join([CHAR_SETS[i] for i in index]))
  return texts


def input_data(image_dir):
  if not gfile.Exists(image_dir):
    print(">> Image director '" + image_dir + "' not found.")
    return None
  extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
  print(">> Looking for images in '" + image_dir + "'")
  file_list = []
  for extension in extensions:
    file_glob = os.path.join(image_dir, '*.' + extension)
    file_list.extend(gfile.Glob(file_glob))
  if not file_list:
    print(">> No files found in '" + image_dir + "'")
    return None
  batch_size = len(file_list)
  images = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH], dtype='float32')
  files = []
  i = 0
  for file_name in file_list:
    image = Image.open(file_name)
    image_gray = image.convert('L')
    image_resize = image_gray.resize(size=(IMAGE_WIDTH,IMAGE_HEIGHT))
    image.close()
    input_img = np.array(image_resize, dtype='float32')
    input_img = np.multiply(input_img.flatten(), 1./255) - 0.5    
    images[i,:] = input_img
    base_name = os.path.basename(file_name)
    files.append(base_name)
    i += 1
  return images, files


def run_predict():
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    input_images, input_filenames = input_data(FLAGS.captcha_dir)
    images = tf.constant(input_images)
    logits = captcha.inference(images, keep_prob=1)
    result = captcha.output(logits)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    print(tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    recog_result = sess.run(result)
    sess.close()
    text = one_hot_to_texts(recog_result)
    total_count = len(input_filenames)
    true_count = 0.
    for i in range(total_count):
      print('image ' + input_filenames[i] + " precise ----> '" + text[i] + "'")
      if text[i] in input_filenames[i]:
        true_count += 1
    precision = true_count / total_count
    print('%s true/total: %d/%d precision @ 1 = %.3f'
                    %(datetime.now(), true_count, total_count, precision))


def main(_):
  run_predict()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default='./captcha_train',
      help='Directory where to restore checkpoint.'
  )
  parser.add_argument(
      '--captcha_dir',
      type=str,
      default='./data/test_data',
      help='Directory where to get captcha images.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
