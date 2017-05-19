from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import argparse
import sys

import numpy as np
from six.moves import xrange 
import tensorflow as tf
import captcha_model as captcha

FLAGS = None

def tower_loss(scope, keep_prob):
  images, labels = captcha.inputs(train=True, batch_size=FLAGS.batch_size)
  logits = captcha.inference(images, keep_prob)
  _ = captcha.loss(logits, labels)
  losses = tf.get_collection('losses', scope)
  total_loss = tf.add_n(losses, name='total_loss')
  return total_loss


def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def run_train():
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    opt = tf.train.AdamOptimizer(1e-4)
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('tower_%d' % (i)) as scope:
            loss = tower_loss(scope, keep_prob=0.5)
            tf.get_variable_scope().reuse_variables()
            grads = opt.compute_gradients(loss)
            tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    train_op = opt.apply_gradients(grads)
    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True))
    
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('>> %s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        print('>> %s Saving in %s' % (datetime.now(), FLAGS.checkpoint))
        saver.save(sess, FLAGS.checkpoint, global_step=step)
   
        
def main(_):
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  run_train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Batch size.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=1000000,
      help='Number of batches to run.'
  )
  parser.add_argument(
      '--num_gpus',
      type=int,
      default=8,
      help='How many GPUs to use.'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='./captcha_train',
      help='Directory where to write event logs.'
  )
  parser.add_argument(
      '--checkpoint',
      type=str,
      default='./captcha_train/captcha',
      help='Directory where to write checkpoint.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
