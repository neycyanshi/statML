#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os

import tensorflow as tf
from tensorflow.contrib import layers
from six.moves import range
import numpy as np
import zhusuan as zs

from matplotlib import pyplot as plt

def main():
  np.random.seed(654) # 324
  tf.set_random_seed(5)

  # data
  n_x = 2  # D

  # Define model parameters
  n_z = 3  # K

  @zs.reuse('model')
  def gmm(observed, n, n_x, n_z):
    with zs.BayesianNet(observed=observed) as model:
      log_pi = tf.get_variable('log_pi', n_z, initializer=tf.truncated_normal_initializer(mean=1., stddev=0.1))
      mu = tf.get_variable('mu', [n_x, n_z], initializer=tf.random_uniform_initializer(-10,10))
      log_sigma = tf.get_variable('log_sigma', [n_x, n_z], initializer=tf.truncated_normal_initializer(stddev=0.2))
      z = zs.OnehotCategorical('z', log_pi, n_samples=n)
      x_mean = tf.matmul(tf.to_float(z.tensor), tf.transpose(mu))
      x_logstd = tf.matmul(tf.to_float(z.tensor), tf.transpose(log_sigma))
      x = zs.Normal('x', x_mean, x_logstd, group_event_ndims=1)
    return model, x.tensor, z.tensor

  @zs.reuse('variational')
  def q_net(x, n_x, n_z, n_hidden=128):
    with zs.BayesianNet() as variational:
      lz_x = layers.fully_connected(tf.to_float(x), n_hidden)
      lz_x = layers.fully_connected(lz_x, n_hidden)
      log_z_pi = layers.fully_connected(lz_x, n_z, activation_fn=None)
      z = zs.OnehotCategorical('z', log_z_pi)
    return variational

  x = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
  n = tf.shape(x)[0]

  def log_joint(observed):
    model, _, _ = gmm(observed, n, n_x, n_z)
    log_pz, log_px_z = model.local_log_prob(['z', 'x'])
    return log_pz + log_px_z

  variational = q_net(x, n_x, n_z) # see what is variational.output??
  qz_samples, log_qz = variational.query('z', outputs=True,
                                         local_log_prob=True)
  surrogate, vlb = zs.nvil(log_joint,
                           observed={'x': x},
                           latent={'z': [qz_samples, log_qz]},
                           axis=0)
  # loss = tf.reduce_mean(surrogate - vlb)
  loss = surrogate

  optimizer = tf.train.AdamOptimizer(0.01)
  infer = optimizer.minimize(loss)

  # Generate mixture of 2-D guassian
  LOG_PI = np.array([1, 1.1, 1.2])
  MU = np.array([[-5, 0, 5], [-4, 4, -4]]) # (n_x, n_z)
  LOG_SIGMA = np.array([[0, 0, 0], [0, 0, 0]])
  with tf.variable_scope('model', reuse=True):
    log_pi = tf.get_variable('log_pi')
    mu = tf.get_variable('mu')
    log_sigma = tf.get_variable('log_sigma')
  log_pi_assign = log_pi.assign(LOG_PI)
  mu_assign = mu.assign(MU)
  log_sigma_assign = log_sigma.assign(LOG_SIGMA)

  n_gen = 100 # N
  _, x_gen, z_gen = gmm({}, n_gen, n_x, n_z)

  # Define training parameters
  # TODO: add batch operation N in gmm
  epochs = 1000
  # save_freq = 1
  # print([var.name for var in tf.trainable_variables()])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # generate the data first
    # sess.run([log_pi_assign, mu_assign, log_sigma_assign])
    log_pi_new, mu_new, log_sigma_new = sess.run([log_pi, mu, log_sigma])
    print('True params in gmm:\nlog_pi:\n{}\nmu:\n{}\nlog_sigma:\n{}\n'\
          .format(log_pi_new, mu_new, log_sigma_new))
    x_train, z_train = sess.run([x_gen, z_gen])
    cls_train = np.dot(z_train, np.array([0,1,2])) # cluster_id (N, 1)

    # train
    sess.run(tf.global_variables_initializer()) # reset model parmas to random
    log_pi_new, mu_new, log_sigma_new = sess.run([log_pi, mu, log_sigma])
    print('Random initial params in gmm:\nlog_pi:\n{}\nmu:\n{}\nlog_sigma:\n{}\n'\
          .format(log_pi_new, mu_new, log_sigma_new))
    for epoch in range(epochs):
      np.random.shuffle([x_train, cls_train])
      lbs = [] # TODO: visualize lbs
      _, lb = sess.run([infer, vlb], feed_dict={x: x_train})
      lbs.append(lb)

      if epoch % 50 == 0:
        print('Epoch {}:\tvlb= {}'.format(epoch, np.mean(lb)))

    print('training completed!\n')
    log_pi_new, mu_new, log_sigma_new = sess.run([log_pi, mu, log_sigma])
    print('Estimated params in gmm:\nlog_pi:\n{}\nmu:\n{}\nlog_sigma:\n{}\n'\
          .format(log_pi_new, mu_new, log_sigma_new))

    qz = sess.run(qz_samples, feed_dict={x: x_train}) # TODO: can I get directly from last ouput of q_net??
    cls_pred = np.dot(qz, np.array([0,1,2]))

    color = np.array(['c', 'm', 'y'])
    # color_pred = np.array(['b', 'g', 'r'])
    marker = np.array(['o', 'D', '^'])

    # visualize estimated and original gmm mixture
    # x_new, z_new = sess.run([x_gen, z_gen])
    # cls_new = np.dot(z_new, np.array([0,1,2]))
    for i in range(n_z):
      plt.scatter(x_train[cls_train==i,0], x_train[cls_train==i,1],
                  c=color[cls_pred[cls_train==i]], marker=marker[i], s=50)
      # plt.scatter(x_new[cls_new==i,0], x_new[cls_new==i,1], c=color_pred[i], marker=marker[i], s=50)

    plt.show()


if __name__ == "__main__":
    main()
