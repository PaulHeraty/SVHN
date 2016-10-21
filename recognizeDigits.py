#!/usr/bin/python

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
import scipy.io as sio
from six.moves.urllib.request import urlretrieve
import h5py

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
  
def eval_accuracy(eval_l_preds, eval_preds, l_labels, labels, masks):
    concatted = np.concatenate((np.reshape((eval_l_preds == l_labels), [-1, 1]), 
                                (eval_preds * masks) == labels), axis=1)
    return 100.0 * (np.sum([np.all(row) for row in concatted])) / len(labels)
          
# START OF MAIN PROGRAM
image_size = 32
num_labels = 10
num_channels = 1 # grayscale


pickle_file = './svhn.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

# Let's draw a couple of the images to ensure they are read correctly
img_indx = 10
plt.imshow(train_dataset[:,:,:,img_indx])
print (train_labels[img_indx])
plt.show()

qwe

batch_size = 128
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  keep_prob = tf.placeholder(tf.float32)
  
  # Model.
  def model(data):
    conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    pooling = tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv = tf.nn.conv2d(pooling, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    pooling = tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    shape = pooling.get_shape().as_list()
    reshape = tf.reshape(pooling, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    hidden_drop = tf.nn.dropout(hidden, keep_prob)
    return tf.matmul(hidden_drop, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  beta = 0.005
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + 
    beta*tf.nn.l2_loss(layer1_weights) + 
    beta*tf.nn.l2_loss(layer2_weights) +
    beta*tf.nn.l2_loss(layer3_weights) +
    beta*tf.nn.l2_loss(layer4_weights) +
    beta*tf.nn.l2_loss(layer1_biases) +
    beta*tf.nn.l2_loss(layer2_biases) + 
    beta*tf.nn.l2_loss(layer3_biases) +
    beta*tf.nn.l2_loss(layer4_biases))
    
  # Optimizer.
  global_step = tf.Variable(0) # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.05, global_step, 3500, 0.96, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 10001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 0.7}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 1.0}), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(feed_dict={keep_prob:1.0}), test_labels))
