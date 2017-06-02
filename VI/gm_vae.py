# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time
import numpy as np
from six.moves import range

import tensorflow as tf
from tensorflow.contrib import layers
import zhusuan as zs
from zhusuan.utils import merge_dicts

import dataset
from utils import save_image_collections


@zs.reuse('model')
def vae(observed, batch_size, n_x, n_h, n_z, n_particles):
  with zs.BayesianNet(observed=observed) as model:
    log_pi = tf.get_variable('log_pi', n_z, initializer=tf.zeros_initializer())
    mu = tf.get_variable('mu', [n_h, n_z], initializer=tf.random_uniform_initializer(-1,1))
    log_sigma = tf.get_variable('log_sigma', [n_h, n_z], initializer=tf.random_normal_initializer(0, 0.1))

    n_log_pi = tf.tile(tf.expand_dims(log_pi, 0), [batch_size, 1]) # (batch_size, n_z)
    z = zs.OnehotCategorical('z', n_log_pi, n_samples=n_particles, group_event_ndims=0) # (n_particles, batch_size, n_z)

    z_tensor = tf.reshape(z.tensor, [-1, n_z])
    h_mean = tf.matmul(tf.to_float(z_tensor), tf.transpose(mu)) # (n_particles x batch_size, n_z) OneHot val_shape [n_z]
    h_logstd = tf.matmul(tf.to_float(z_tensor), tf.transpose(log_sigma))

    h_mean = tf.reshape(h_mean, [-1, batch_size, n_h]) # (n_particles, batch_size, n_h)
    h_logstd = tf.reshape(h_logstd, [-1, batch_size, n_h])

    # returned tensor of log_prob() has shape  (... + )batch_shape[:-group_event_ndims] = (n_particles, batch_size).
    h = zs.Normal('h', h_mean, h_logstd, group_event_ndims=1) # Multivariate Normal. val_shape []. see zhusuan Basic Concepts.
    lx_h = layers.fully_connected(h, 500)
    lx_h = layers.fully_connected(lx_h, 500)
    x_logits = layers.fully_connected(lx_h, n_x, activation_fn=None) # (n_particles, batch_size, n_x)
    x = zs.Bernoulli('x', x_logits, group_event_ndims=1) # (n_particles, batch_size, n_x) n_x=784 pixel as one event


  return model, x_logits, z.tensor

@zs.reuse('variational')
def q_net(x, n_h, n_particles):
  with zs.BayesianNet() as variational:
    lh_x = layers.fully_connected(tf.to_float(x), 500)
    lh_x = layers.fully_connected(lh_x, 500)
    h_mean = layers.fully_connected(lh_x, n_h, activation_fn=None)
    h_logstd = layers.fully_connected(lh_x, n_h, activation_fn=None)
    h = zs.Normal('h', h_mean, h_logstd, n_samples=n_particles, group_event_ndims=1)

  return variational


def save_img(sess, x, z, ckpt_path, epoch):
  x_val, z_val = sess.run([x, z])
  img_path = os.path.join(ckpt_path, "result.epoch{}.png".format(epoch))
  save_image_collections(x_val, img_path)
  return z_val

def print_param(param_name):
  val = sess.run(param_name)
  if 'log' in param_name:
    val = np.exp(val)
  val_normalized = val / np.sum(val)
  print('{}: {}'.format(param_name, val_normalized))

if __name__ == "__main__":
  tf.set_random_seed(666)
  np.random.seed(666)

  # Load data from MNIST
  data_dir = './data'
  data_path = os.path.join(data_dir, 'mnist.pkl.gz')
  x_train, t_train, x_val, t_val, x_test, t_test = dataset.load_mnist_realval(data_path)
  x_train = np.vstack([x_train, x_val]).astype('float32')
  n_x = x_train.shape[1] # 784=28*28

  # Define model parameters
  n_h = 40 # D
  n_z = 10 # K

  # Define training/evaluation parameters
  lb_samples = 10
  epoches = 100
  batch_size = 100
  iters = x_train.shape[0] // batch_size
  learning_rate = 0.001
  save_freq = 20
  ckpt_path = "./ckpt/10x10_2"

  # Computation graph of model-inference-learning
  n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
  x_orig = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
  x_bin = tf.cast(tf.less(tf.random_uniform(tf.shape(x_orig), 0, 1), x_orig), tf.int32)
  x = tf.placeholder(tf.int32, shape=[None, n_x], name='x')
  x_obs = tf.tile(tf.expand_dims(x, 0), [n_particles, 1, 1]) # (n_particles, batch_size, n_x)
  n = tf.shape(x)[0] # batch_size
  z_obs = tf.placeholder(tf.int32, shape=[n_z, None, None]) # (n_z, n_particles, batch_size)

  def log_joint(observed):
    log_pz = []
    log_ph_z = []
    log_px_h = None
    for i in range(n_z):
      z_obs_onehot = tf.one_hot(i * tf.ones([n_particles, batch_size], dtype=tf.int32), depth=n_z, dtype=tf.int32) # (n_particles, batch_size, n_z)
      ob_dict = merge_dicts(observed, {'z': z_obs_onehot}) # sum over all possible z
      model, _, _ = vae(ob_dict, n, n_x, n_h, n_z, n_particles)
      log_pz_i, log_ph_z_i, log_px_h = model.local_log_prob(['z', 'h', 'x'])
      log_pz.append(log_pz_i)
      log_ph_z.append(log_ph_z_i)
    log_pz = tf.stack(log_pz, axis=-1)
    log_ph_z = tf.stack(log_ph_z, axis=-1)
    # P(X,H) = P(X|H) * sum[(P(H|z_i) * P(z_i))]
    return log_px_h + tf.reduce_logsumexp(log_pz+log_ph_z, axis=-1)

  variational = q_net(x, n_h, n_particles)
  qh_samples, log_qh = variational.query('h', outputs=True, local_log_prob=True)
  lower_bound = tf.reduce_mean(zs.sgvb(log_joint, {'x': x_obs}, {'h': [qh_samples, log_qh]}, axis=0))

  optimizer = tf.train.AdamOptimizer(learning_rate)
  infer = optimizer.minimize(-lower_bound)

  # Computation graph of generating images
  n_gen = 100
  n_per_class = n_gen // n_z
  z_gen = np.zeros([n_gen, n_z])
  for i in range(n_z):
    z_gen[i * n_per_class:(i+1) * n_per_class, i] = 1
  z_gen = np.expand_dims(z_gen, axis=0)
  _, x_logits, z_onehot = vae({'z':z_gen}, n_gen, n_x, n_h, n_z, n_particles=1)
  # _, x_logits, z_onehot = vae({}, n_gen, n_x, n_h, n_z, n_particles=1)
  x_gen = tf.reshape(tf.sigmoid(x_logits), [-1, 28, 28, 1]) # (n_gen, 28, 28, 1)
  z_gt = tf.where(tf.equal(z_onehot, 1))[:,2] # z_onehot shape (n_particles, n_gen, n_z)

  print('trainable_variables:')
  params = tf.trainable_variables()
  for i in params:
    print(i.name, i.get_shape())

  saver = tf.train.Saver(max_to_keep=10)


  # Run the inference
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Restore from the latest checkpoint
    ckpt_file = tf.train.latest_checkpoint(ckpt_path)
    begin_epoch = 1
    if ckpt_file is not None:
      print('Restoring model from {}...'.format(ckpt_file))
      begin_epoch = int(ckpt_file.split('.')[-2]) + 1
      saver.restore(sess, ckpt_file)

    for epoch in range(begin_epoch, epoches + 1):
      time_epoch = -time.time()
      np.random.shuffle(x_train)
      lbs = []
      for t in range(iters):
        x_batch = x_train[t * batch_size:(t + 1) * batch_size]
        x_batch_bin = sess.run(x_bin, feed_dict={x_orig: x_batch})
        _, lb = sess.run([infer, lower_bound],
                         feed_dict={x: x_batch_bin,
                                    n_particles: lb_samples})
        lbs.append(lb)
      time_epoch += time.time()
      print('Epoch {} ({:.1f}s): Lower bound = {}'.format(epoch, time_epoch, np.mean(lbs)))
      # print_param('model/log_pi:0') # check log_pi_val

      if epoch % save_freq == 0:
        print('Saving model and result img...')
        save_path = os.path.join(ckpt_path, "vae.epoch.{}.ckpt".format(epoch))
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        saver.save(sess, save_path)
        _ = save_img(sess, x_gen, z_gt, ckpt_path, epoch)

    print('training done.')
    print_param('model/log_pi:0')

    # save visualization of the results
    z_gt_val = save_img(sess, x_gen, z_gt, ckpt_path, "final")
    print('z_gt:\n%r' % np.reshape(z_gt_val, [n_z, n_per_class]))
