from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import argparse
import sys
import math

import tensorflow as tf
import captcha_model as captcha

FLAGS = None

def run_eval():
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    images, labels = captcha.inputs(train=False, batch_size=FLAGS.batch_size)
    logits = captcha.inference(images, keep_prob=1)
    eval_correct = captcha.evaluation(logits, labels)  
    sess = tf.Session()    
    saver = tf.train.Saver()    
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0
      total_true_count = 0
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      print('>> loop: %d, total_sample_count: %d' % (num_iter, total_sample_count))
      while step < num_iter and not coord.should_stop():
        true_count = sess.run(eval_correct)
        total_true_count += true_count
        precision = true_count / FLAGS.batch_size
        print('>> %s Step %d: true/total: %d/%d precision @ 1 = %.3f'
                    %(datetime.now(), step, true_count, FLAGS.batch_size, precision))
        step += 1
      precision = total_true_count / total_sample_count
      print('>> %s true/total: %d/%d precision @ 1 = %.3f'
                    %(datetime.now(), total_true_count, total_sample_count, precision))       
    except Exception as e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
    coord.join(threads)
    sess.close()


def main(_):
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  run_eval()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--num_examples',
      type=int,
      default=20000,
      help='Number of examples to run validation.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.'
  )
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default='./captcha_train',
      help='Directory where to restore checkpoint.'
  )
  parser.add_argument(
      '--eval_dir',
      type=str,
      default='./captcha_eval',
      help='Directory where to write event logs.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
