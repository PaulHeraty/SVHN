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
import time
#from numpy.oldnumeric.compat import pickle_array

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
  
def eval_accuracy(eval_l_preds, eval_preds, l_labels, labels, masks):
    concatted = np.concatenate((np.reshape((eval_l_preds == l_labels), [-1, 1]), 
                                (eval_preds * masks) == labels), axis=1)
    return 100.0 * (np.sum([np.all(row) for row in concatted])) / len(labels)

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_sizeX, image_sizeY, num_channels)).astype(np.float32)
  digit_number = labels[:,0]
  num_digits_encoded = (np.arange(num_digits) == digit_number[:,None]).astype(np.float32)
  digit1 = labels[:,1]
  digit1_encoded = (np.arange(num_labels) == digit1[:,None]).astype(np.float32)
  digit2 = labels[:,2]
  digit2_encoded = (np.arange(num_labels) == digit2[:,None]).astype(np.float32)
  digit3 = labels[:,3]
  digit3_encoded = (np.arange(num_labels) == digit3[:,None]).astype(np.float32)
  digit4 = labels[:,4]
  digit4_encoded = (np.arange(num_labels) == digit4[:,None]).astype(np.float32)
  digit5 = labels[:,5]
  digit5_encoded = (np.arange(num_labels) == digit5[:,None]).astype(np.float32)

  labels = np.hstack((num_digits_encoded, digit1_encoded, digit2_encoded, digit3_encoded, digit4_encoded, digit5_encoded))
  return dataset, labels

def reformat1(dataset, labels):
  dataset = dataset.reshape((-1, image_sizeX * image_sizeY * num_channels)).astype(np.float32)

  digit_number = labels[:,0]
  num_digits_encoded = (np.arange(num_digits) == digit_number[:,None]).astype(np.float32)
  digit1 = labels[:,1]
  digit1_encoded = (np.arange(num_labels) == digit1[:,None]).astype(np.float32)
  digit2 = labels[:,2]
  digit2_encoded = (np.arange(num_labels) == digit2[:,None]).astype(np.float32)
  digit3 = labels[:,3]
  digit3_encoded = (np.arange(num_labels) == digit3[:,None]).astype(np.float32)
  digit4 = labels[:,4]
  digit4_encoded = (np.arange(num_labels) == digit4[:,None]).astype(np.float32)
  digit5 = labels[:,5]
  digit5_encoded = (np.arange(num_labels) == digit5[:,None]).astype(np.float32)

  labels = np.hstack((num_digits_encoded, digit1_encoded, digit2_encoded, digit3_encoded, digit4_encoded, digit5_encoded))
  return dataset, labels  
            
# START OF MAIN PROGRAM
debug = 0
graph_type = 1
image_sizeX = 32
image_sizeY = 32
num_channels = 3 # rgb
num_digits = 5
num_labels = 10

pickle_file = './svhn.pickle'

# Start by reading in the pickle datasets
print("Reading pickle file {}".format(pickle_file))
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

if debug:
    # Let's draw one of the images to ensure they are read correctly
    img_index = 1000
    plt.imshow(train_dataset[img_index,:,:,:])
    print (train_labels[img_index])
    plt.show()

batch_size = 128
patch_size = 5
depth = 16
num_hidden = 64

# Single layer NN graph for simple testing
if graph_type == 1:    
    #train_subset = 1000
    train_subset = len(train_dataset)
    if debug:
        print("Number of training samples : {}". format(train_subset))
        
    # Reformat to encode labels etc.
    # labels is not of the format (one_shot_enc_num_digits, one_shot_enc_digit1, ... , one_shot_enc_digit5)
    train_dataset, train_labels = reformat1(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat1(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat1(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    
    graph = tf.Graph()
    with graph.as_default():
    
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
      
        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random values following a (truncated)
        # normal distribution. The biases get initialized to zero.
        weights_num_digits = tf.Variable(tf.truncated_normal([image_sizeX * image_sizeY * num_channels, num_digits]))
        weights_digit1 = tf.Variable(tf.truncated_normal([image_sizeX * image_sizeY * num_channels, num_labels]))
        weights_digit2 = tf.Variable(tf.truncated_normal([image_sizeX * image_sizeY * num_channels, num_labels]))
        weights_digit3 = tf.Variable(tf.truncated_normal([image_sizeX * image_sizeY * num_channels, num_labels]))
        weights_digit4 = tf.Variable(tf.truncated_normal([image_sizeX * image_sizeY * num_channels, num_labels]))
        weights_digit5 = tf.Variable(tf.truncated_normal([image_sizeX * image_sizeY * num_channels, num_labels]))
        biases_num_digits = tf.Variable(tf.zeros([num_digits]))
        biases_digit1 = tf.Variable(tf.zeros([num_labels]))
        biases_digit2 = tf.Variable(tf.zeros([num_labels]))
        biases_digit3 = tf.Variable(tf.zeros([num_labels]))
        biases_digit4 = tf.Variable(tf.zeros([num_labels]))
        biases_digit5 = tf.Variable(tf.zeros([num_labels]))
      
        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        logits_num_digits = tf.matmul(tf_train_dataset, weights_num_digits) + biases_num_digits
        logits_digit1 = tf.matmul(tf_train_dataset, weights_digit1) + biases_digit1
        logits_digit2 = tf.matmul(tf_train_dataset, weights_digit2) + biases_digit2
        logits_digit3 = tf.matmul(tf_train_dataset, weights_digit3) + biases_digit3
        logits_digit4 = tf.matmul(tf_train_dataset, weights_digit4) + biases_digit4
        logits_digit5 = tf.matmul(tf_train_dataset, weights_digit5) + biases_digit5
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_num_digits, tf_train_labels[:,0:5]) +
                              tf.nn.softmax_cross_entropy_with_logits(logits_digit1, tf_train_labels[:,5:15]) +
                              tf.nn.softmax_cross_entropy_with_logits(logits_digit2, tf_train_labels[:,15:25]) +
                              tf.nn.softmax_cross_entropy_with_logits(logits_digit3, tf_train_labels[:,25:35]) +
                              tf.nn.softmax_cross_entropy_with_logits(logits_digit4, tf_train_labels[:,35:45]) +
                              tf.nn.softmax_cross_entropy_with_logits(logits_digit5, tf_train_labels[:,45:55]))
      
        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
      
        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction_num_digits = tf.nn.softmax(logits_num_digits)
        train_prediction_digit1 = tf.nn.softmax(logits_digit1)
        train_prediction_digit2 = tf.nn.softmax(logits_digit2)
        train_prediction_digit3 = tf.nn.softmax(logits_digit3)
        train_prediction_digit4 = tf.nn.softmax(logits_digit4)
        train_prediction_digit5 = tf.nn.softmax(logits_digit5)
        valid_prediction_num_digits = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights_num_digits) + biases_num_digits)
        valid_prediction_digit1 = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights_digit1) + biases_digit1)
        valid_prediction_digit2 = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights_digit2) + biases_digit2)
        valid_prediction_digit3 = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights_digit3) + biases_digit3)
        valid_prediction_digit4 = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights_digit4) + biases_digit4)
        valid_prediction_digit5 = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights_digit5) + biases_digit5)
        test_prediction_num_digits = tf.nn.softmax(tf.matmul(tf_test_dataset, weights_num_digits) + biases_num_digits)
        test_prediction_digit1 = tf.nn.softmax(tf.matmul(tf_test_dataset, weights_digit1) + biases_digit1)
        test_prediction_digit2 = tf.nn.softmax(tf.matmul(tf_test_dataset, weights_digit2) + biases_digit2)
        test_prediction_digit3 = tf.nn.softmax(tf.matmul(tf_test_dataset, weights_digit3) + biases_digit3)
        test_prediction_digit4 = tf.nn.softmax(tf.matmul(tf_test_dataset, weights_digit4) + biases_digit1)
        test_prediction_digit5 = tf.nn.softmax(tf.matmul(tf_test_dataset, weights_digit5) + biases_digit5)
      
        num_steps = 1000

        def accuracy(predictions, labels):
            ac_nd = 1.0 * np.sum(np.argmax(predictions[0], 1) == np.argmax(labels[:,0:5], 1)) / predictions[0].shape[0]
            ac_d1 = 1.0 * np.sum(np.argmax(predictions[1], 1) == np.argmax(labels[:,5:15], 1)) / predictions[1].shape[0]
            ac_d2 = 1.0 * np.sum(np.argmax(predictions[2], 1) == np.argmax(labels[:,15:25], 1)) / predictions[2].shape[0]
            ac_d3 = 1.0 * np.sum(np.argmax(predictions[3], 1) == np.argmax(labels[:,25:35], 1)) / predictions[3].shape[0]
            ac_d4 = 1.0 * np.sum(np.argmax(predictions[4], 1) == np.argmax(labels[:,35:45], 1)) / predictions[4].shape[0]
            ac_d5 = 1.0 * np.sum(np.argmax(predictions[5], 1) == np.argmax(labels[:,45:55], 1)) / predictions[5].shape[0]
            overall = ac_nd * ac_d1 * ac_d2 * ac_d3 * ac_d4 * ac_d5
            return ac_nd, ac_d1, ac_d2, ac_d3, ac_d4, ac_d5, overall
            
        start = time.time()
        with tf.Session(graph=graph) as session:
            # This is a one-time operation which ensures the parameters get initialized as
            # we described in the graph: random weights for the matrix, zeros for the biases. 
             tf.initialize_all_variables().run()
             print('Initialized')
             for step in range(num_steps):
                 # Run the computations. We tell .run() that we want to run the optimizer,
                 # and get the loss value and the training predictions returned as numpy
                 # arrays.
                 _, l, prediction_num_digits, prediction_digit1, prediction_digit2, prediction_digit3, prediction_digit4, prediction_digit5 = session.run([optimizer, loss, train_prediction_num_digits, 
                                                  train_prediction_digit1, train_prediction_digit2, 
                                                  train_prediction_digit3, train_prediction_digit4, train_prediction_digit5])
                 if (step % 100 == 0):
                     print("Loss at step {}: {}".format(step, l))
                     print("Prediction num digits")
                     print('Training accuracy: {}'.format(accuracy(
                            (prediction_num_digits, prediction_digit1, prediction_digit2, 
                            prediction_digit3, prediction_digit4, prediction_digit5), train_labels[:train_subset, :])))
                     # Calling .eval() on valid_prediction is basically like calling run(), but
                     # just to get that one numpy array. Note that it recomputes all its graph
                     # dependencies.
                     print('Validation accuracy: {}'.format(accuracy(
                            (valid_prediction_num_digits.eval(),
                            valid_prediction_digit1.eval(), valid_prediction_digit2.eval(), valid_prediction_digit3.eval(),
                            valid_prediction_digit4.eval(), valid_prediction_digit5.eval()), valid_labels)))
             print('Test accuracy: {}'.format(accuracy((test_prediction_num_digits.eval(), 
                    test_prediction_digit1.eval(), test_prediction_digit2.eval(), test_prediction_digit3.eval(),
                    test_prediction_digit4.eval(), test_prediction_digit5.eval()),test_labels)))
             end = time.time()
             print("Time taken to train database : {} seconds".format(end - start))
# 
# Reformat to encode labels etc.
# labels is not of the format (one_shot_enc_num_digits, one_shot_enc_digit1, ... , one_shot_enc_digit5)
#train_dataset, train_labels = reformat(train_dataset, train_labels)
#valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
#test_dataset, test_labels = reformat(test_dataset, test_labels)
#print('Training set', train_dataset.shape, train_labels.shape)
#print('Validation set', valid_dataset.shape, valid_labels.shape)
#print('Test set', test_dataset.shape, test_labels.shape)
#
# with graph.as_default():
# 
#   # Input data.
#   tf_train_dataset = tf.placeholder(
#     tf.float32, shape=(batch_size, image_size, image_size, num_channels))
#   tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#   tf_valid_dataset = tf.constant(valid_dataset)
#   tf_test_dataset = tf.constant(test_dataset)
#   
#   # Variables.
#   layer1_weights = tf.Variable(tf.truncated_normal(
#       [patch_size, patch_size, num_channels, depth], stddev=0.1))
#   layer1_biases = tf.Variable(tf.zeros([depth]))
#   layer2_weights = tf.Variable(tf.truncated_normal(
#       [patch_size, patch_size, depth, depth], stddev=0.1))
#   layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
#   layer3_weights = tf.Variable(tf.truncated_normal(
#       [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
#   layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
#   layer4_weights = tf.Variable(tf.truncated_normal(
#       [num_hidden, num_labels], stddev=0.1))
#   layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
#   keep_prob = tf.placeholder(tf.float32)
#   
#   # Model.
#   def model(data):
#     conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer1_biases)
#     pooling = tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#     conv = tf.nn.conv2d(pooling, layer2_weights, [1, 1, 1, 1], padding='SAME')
#     hidden = tf.nn.relu(conv + layer2_biases)
#     pooling = tf.nn.max_pool(hidden, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#     shape = pooling.get_shape().as_list()
#     reshape = tf.reshape(pooling, [shape[0], shape[1] * shape[2] * shape[3]])
#     hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
#     hidden_drop = tf.nn.dropout(hidden, keep_prob)
#     return tf.matmul(hidden_drop, layer4_weights) + layer4_biases
#   
#   # Training computation.
#   logits = model(tf_train_dataset)
#   #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
#   beta = 0.005
#   loss = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + 
#     beta*tf.nn.l2_loss(layer1_weights) + 
#     beta*tf.nn.l2_loss(layer2_weights) +
#     beta*tf.nn.l2_loss(layer3_weights) +
#     beta*tf.nn.l2_loss(layer4_weights) +
#     beta*tf.nn.l2_loss(layer1_biases) +
#     beta*tf.nn.l2_loss(layer2_biases) + 
#     beta*tf.nn.l2_loss(layer3_biases) +
#     beta*tf.nn.l2_loss(layer4_biases))
#     
#   # Optimizer.
#   global_step = tf.Variable(0) # count the number of steps taken.
#   learning_rate = tf.train.exponential_decay(0.05, global_step, 3500, 0.96, staircase=True)
#   optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#   
#   # Predictions for the training, validation, and test data.
#   train_prediction = tf.nn.softmax(logits)
#   valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
#   test_prediction = tf.nn.softmax(model(tf_test_dataset))
# 
# num_steps = 10001
# 
# with tf.Session(graph=graph) as session:
#   tf.initialize_all_variables().run()
#   print('Initialized')
#   for step in range(num_steps):
#     offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
#     batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
#     batch_labels = train_labels[offset:(offset + batch_size), :]
#     feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 0.7}
#     _, l, predictions = session.run(
#       [optimizer, loss, train_prediction], feed_dict=feed_dict)
#     if (step % 500 == 0):
#       print('Minibatch loss at step %d: %f' % (step, l))
#       print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
#       print('Validation accuracy: %.1f%%' % accuracy(
#         valid_prediction.eval(feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 1.0}), valid_labels))
#   print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(feed_dict={keep_prob:1.0}), test_labels))
