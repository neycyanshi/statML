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
from matplotlib.patches import Ellipse
from utils import var_regularizer, l1_regularizer

def main():
  manual_seed = 6
  np.random.seed(manual_seed)
  tf.set_random_seed(manual_seed)

  # data
  n_x = 2  # D

  # Define model parameters
  n_z = 3  # K

  @zs.reuse('model')
  def gmm(observed, n, n_x, n_z):
    with zs.BayesianNet(observed=observed) as model:
      log_pi = tf.get_variable('log_pi', n_z, initializer=tf.truncated_normal_initializer(mean=1., stddev=0.5),
                               regularizer=var_regularizer(1.0))
      mu = tf.get_variable('mu', [n_x, n_z], initializer=tf.orthogonal_initializer(gain=4.0)) # try uniform init
      log_sigma = tf.get_variable('log_sigma', [n_x, n_z], initializer=tf.truncated_normal_initializer(stddev=0.5),
                                  regularizer=l1_regularizer(0.01)) # try not l1_reg
      z = zs.OnehotCategorical('z', log_pi, n_samples=n)
      x_mean = tf.matmul(tf.to_float(z.tensor), tf.transpose(mu))
      x_logstd = tf.matmul(tf.to_float(z.tensor), tf.transpose(log_sigma))
      x = zs.Normal('x', x_mean, x_logstd, group_event_ndims=1)
    return model, x.tensor, z.tensor

  @zs.reuse('variational')
  def q_net(x, n_x, n_z, n_hidden=8):
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
  # loss = surrogate

  with tf.variable_scope('model', reuse=True):
    log_pi = tf.get_variable('log_pi')
    mu = tf.get_variable('mu')
    log_sigma = tf.get_variable('log_sigma')
  # LOG_PI = np.array([1, 1.1, 1.2])
  # MU = np.array([[-5, 0, 5], [-4, 4, -4]]) # (n_x, n_z)
  # LOG_SIGMA = np.array([[0, 0, 0], [0, 0, 0]])
  # log_pi_assign = log_pi.assign(LOG_PI)
  # mu_assign = mu.assign(MU)
  # log_sigma_assign = log_sigma.assign(LOG_SIGMA)

  reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  reg_loss = sum(reg_losses)
  reg_constant = 5.0
  loss = surrogate + reg_constant * reg_loss

  optimizer = tf.train.AdamOptimizer(0.01)
  infer = optimizer.minimize(loss)

  # Generate mixture of 2-D guassian
  n_gen = 100 # N
  _, x_gen, z_gen = gmm({}, n_gen, n_x, n_z)

  # Define training parameters
  # TODO: add batch operation N in gmm
  epochs = 2000
  # save_freq = 1
  # print([var.name for var in tf.trainable_variables()])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # generate the data first
    # sess.run([log_pi_assign, mu_assign, log_sigma_assign])
    log_pi_val, mu_val, log_sigm_val = sess.run([log_pi, mu, log_sigma])
    print('True params in gmm:\nlog_pi:\n{}\nmu:\n{}\nlog_sigma:\n{}\n'\
          .format(log_pi_val/np.linalg.norm(log_pi_val), mu_val, log_sigm_val))
    x_train, z_train = sess.run([x_gen, z_gen])
    cls_train = np.dot(z_train, np.arange(n_z)) # cluster_id (N, 1)

    # train
    sess.run(tf.global_variables_initializer()) # reset model parmas to random
    log_pi_val, mu_val, log_sigm_val = sess.run([log_pi, mu, log_sigma])
    print('Random initial params in gmm:\nlog_pi:\n{}\nmu:\n{}\nlog_sigma:\n{}\n'\
          .format(log_pi_val/np.linalg.norm(log_pi_val), mu_val, log_sigm_val))
    for epoch in range(epochs):
      np.random.shuffle([x_train, cls_train])
      lbs = [] # TODO: visualize lbs
      _, lb = sess.run([infer, vlb], feed_dict={x: x_train})
      lbs.append(lb)

      if epoch % 50 == 0:
        print('Epoch {}:\tvlb = {}\treg_loss = {}'.format(epoch, np.mean(lb), sess.run(reg_loss)))

    print('training completed!\n')
    log_pi_val, mu_val, log_sigm_val = sess.run([log_pi, mu, log_sigma])
    print('Estimated params in gmm:\nlog_pi:\n{}\nmu:\n{}\nlog_sigma:\n{}\n'\
          .format(log_pi_val/np.linalg.norm(log_pi_val), mu_val, log_sigm_val))

    qz = sess.run(qz_samples, feed_dict={x: x_train})
    cls_pred = np.dot(qz, np.arange(n_z))

    color = np.array(['c', 'm', 'y', 'b', 'g', 'r']) # extend color list if n_z > 6
    marker = np.array(['o', 'D', '^', 'x', 's', 'v'])

    # visualize estimated and original gmm mixture
    # x_new, z_new = sess.run([x_gen, z_gen])
    # cls_new = np.dot(z_new, np.array([0,1,2]))
    fig = plt.figure(num=0, figsize=(6,6))
    ax = fig.add_subplot(111, aspect='equal')
    for i in range(n_z):
      plt.scatter(x_train[cls_train==i,0], x_train[cls_train==i,1],
                  c=color[cls_pred[cls_train==i]], marker=marker[i], s=50)
      ell = Ellipse(xy=mu_val[:,i], width=6*np.exp(log_sigm_val[0,i]), height=6*np.exp(log_sigm_val[1,i]),
                    facecolor='none')
      ax.add_patch(ell)
      # plt.scatter(x_train[cls_train==i,0], x_train[cls_train==i,1], c=color[i], marker=marker[i], s=50)
      # plt.scatter(x_new[cls_new==i,0], x_new[cls_new==i,1], c=color[3+i], marker=marker[i], s=50)

    fig.savefig('N'+str(n_gen)+'_K'+str(n_z)+'_seed'+str(manual_seed)+'.png')
    plt.show()


if __name__ == "__main__":
    main()
