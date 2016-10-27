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

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_sizeX * image_sizeY * num_channels)).astype(np.float32)

  digit_number = labels[:,0]
  num_digits_encoded = (np.arange(num_digits) == digit_number[:,None]).astype(np.float32)
  digit1 = labels[:,1]
  digit1_encoded = (np.arange(num_labels) == digit1[:,None]).astype(np.float32)
  digit2 = labels[:,2]
  digit2_encoded = (np.arange(num_labels) == digit2[:,None]).astype(np.float32)
  digit3 = labels[:,3]
  digit3_encoded = (np.arange(num_labels) == digit3[:,None]).astype(np.float32)

  labels = np.hstack((num_digits_encoded, digit1_encoded, digit2_encoded, digit3_encoded))
  return dataset, labels  
            
# START OF MAIN PROGRAM
debug = 0
graph_type = 1
image_sizeX = 32
image_sizeY = 32
num_channels = 1 # grayscale
num_digits = 3
num_labels = 10

pickle_file = './svhn_3digits_gray.pickle'

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
hidden_layer1_size =  1024
hidden_layer2_size =  512

# Single layer NN graph for simple testing
if graph_type == 1:    
    #train_subset = 1000
    train_subset = len(train_dataset)
    if debug:
        print("Number of training samples : {}". format(train_subset))
        
    # Reformat to encode labels etc.
    # labels is not of the format (one_shot_enc_num_digits, one_shot_enc_digit1, ... , one_shot_enc_digit5)
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    
    graph = tf.Graph()
    with graph.as_default():
    
        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_sizeX * image_sizeY * num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_digits + num_digits * num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)
      
        # Variables.
        weights_num_digits_h1 = tf.Variable(tf.truncated_normal([image_sizeX * image_sizeY * num_channels, hidden_layer1_size]))
        weights_num_digits_h1 = tf.get_variable("weights_num_digits_h1", 
                                                shape = [image_sizeX * image_sizeY * num_channels, hidden_layer1_size], 
                                                initializer=tf.contrib.layers.xavier_initializer())
        weights_num_digits_h2 = tf.Variable(tf.truncated_normal([hidden_layer1_size, hidden_layer2_size]))
        weights_num_digits_h2 = tf.get_variable("weights_num_digits_h2", 
                                                shape = [hidden_layer1_size, hidden_layer2_size], 
                                                initializer=tf.contrib.layers.xavier_initializer())
        weights_num_digits_o = tf.Variable(tf.truncated_normal([hidden_layer2_size, num_digits]))
        weights_num_digits_o = tf.get_variable("weights_num_digits_o", 
                                                shape = [hidden_layer2_size, num_digits], 
                                                initializer=tf.contrib.layers.xavier_initializer())
        weights_digit1_h1 = tf.Variable(tf.truncated_normal([image_sizeX * image_sizeY * num_channels, hidden_layer1_size]))
        weights_digit1_h1 = tf.get_variable("weights_digit1_h1", 
                                                shape = [image_sizeX * image_sizeY * num_channels, hidden_layer1_size], 
                                                initializer=tf.contrib.layers.xavier_initializer())
        weights_digit1_h2 = tf.Variable(tf.truncated_normal([hidden_layer1_size, hidden_layer2_size]))
        weights_digit1_h2 = tf.get_variable("weights_digit1_h2", 
                                                shape = [hidden_layer1_size, hidden_layer2_size], 
                                                initializer=tf.contrib.layers.xavier_initializer())
        weights_digit1_o = tf.Variable(tf.truncated_normal([hidden_layer2_size, num_labels]))
        weights_digit1_o = tf.get_variable("weights_digit1_o", 
                                                shape = [hidden_layer2_size, num_labels], 
                                                initializer=tf.contrib.layers.xavier_initializer())
        weights_digit2_h1 = tf.Variable(tf.truncated_normal([image_sizeX * image_sizeY * num_channels, hidden_layer1_size]))
        weights_digit2_h1 = tf.get_variable("weights_digit2_h1", 
                                                shape = [image_sizeX * image_sizeY * num_channels, hidden_layer1_size], 
                                                initializer=tf.contrib.layers.xavier_initializer())
        weights_digit2_h2 = tf.Variable(tf.truncated_normal([hidden_layer1_size, hidden_layer2_size]))
        weights_digit2_h2 = tf.get_variable("weights_digit2_h2", 
                                                shape = [hidden_layer1_size, hidden_layer2_size], 
                                                initializer=tf.contrib.layers.xavier_initializer())
        weights_digit2_o = tf.Variable(tf.truncated_normal([hidden_layer2_size, num_labels]))
        weights_digit2_o = tf.get_variable("weights_digit2_o", 
                                                shape = [hidden_layer2_size, num_labels], 
                                                initializer=tf.contrib.layers.xavier_initializer())
        weights_digit3_h1 = tf.Variable(tf.truncated_normal([image_sizeX * image_sizeY * num_channels, hidden_layer1_size]))
        weights_digit3_h1 = tf.get_variable("weights_digit3_h1", 
                                                shape = [image_sizeX * image_sizeY * num_channels, hidden_layer1_size], 
                                                initializer=tf.contrib.layers.xavier_initializer())
        weights_digit3_h2 = tf.Variable(tf.truncated_normal([hidden_layer1_size, hidden_layer2_size]))
        weights_digit3_h2 = tf.get_variable("weights_digit3_h2", 
                                                shape = [hidden_layer1_size, hidden_layer2_size], 
                                                initializer=tf.contrib.layers.xavier_initializer())
        weights_digit3_o = tf.Variable(tf.truncated_normal([hidden_layer2_size, num_labels]))
        weights_digit3_o = tf.get_variable("weights_digit3_o", 
                                                shape = [hidden_layer2_size, num_labels], 
                                                initializer=tf.contrib.layers.xavier_initializer())

        biases_num_digits_h1 = tf.Variable(tf.zeros([hidden_layer1_size]))
        biases_num_digits_h2 = tf.Variable(tf.zeros([hidden_layer2_size]))
        biases_num_digits_o = tf.Variable(tf.zeros([num_digits]))
        biases_digit1_h1 = tf.Variable(tf.zeros([hidden_layer1_size]))
        biases_digit1_h2 = tf.Variable(tf.zeros([hidden_layer2_size]))
        biases_digit1_o = tf.Variable(tf.zeros([num_labels]))
        biases_digit2_h1 = tf.Variable(tf.zeros([hidden_layer1_size]))
        biases_digit2_h2 = tf.Variable(tf.zeros([hidden_layer2_size]))
        biases_digit2_o = tf.Variable(tf.zeros([num_labels]))
        biases_digit3_h1 = tf.Variable(tf.zeros([hidden_layer1_size]))
        biases_digit3_h2 = tf.Variable(tf.zeros([hidden_layer2_size]))
        biases_digit3_o = tf.Variable(tf.zeros([num_labels]))
        
        keep_prob = tf.placeholder(tf.float32)

        # Training computation.
        hidden1_num_digits = tf.nn.relu(tf.matmul(tf_train_dataset, weights_num_digits_h1) + biases_num_digits_h1)
        hidden1_drop_num_digits = tf.nn.dropout(hidden1_num_digits, keep_prob)
        hidden2_num_digits = tf.nn.relu(tf.matmul(hidden1_drop_num_digits, weights_num_digits_h2) + biases_num_digits_h2)
        hidden2_drop_num_digits = tf.nn.dropout(hidden2_num_digits, keep_prob)
        logits_num_digits = tf.matmul(hidden2_drop_num_digits, weights_num_digits_o) + biases_num_digits_o
        
        hidden1_digit1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_digit1_h1) + biases_digit1_h1)
        hidden1_drop_digit1 = tf.nn.dropout(hidden1_digit1, keep_prob)
        hidden2_digit1 = tf.nn.relu(tf.matmul(hidden1_drop_digit1, weights_digit1_h2) + biases_digit1_h2)
        hidden2_drop_digit1 = tf.nn.dropout(hidden2_digit1, keep_prob)
        logits_digit1 = tf.matmul(hidden2_drop_digit1, weights_digit1_o) + biases_digit1_o
        
        hidden1_digit2 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_digit2_h1) + biases_digit2_h1)
        hidden1_drop_digit2 = tf.nn.dropout(hidden1_digit2, keep_prob)
        hidden2_digit2 = tf.nn.relu(tf.matmul(hidden1_drop_digit2, weights_digit2_h2) + biases_digit2_h2)
        hidden2_drop_digit2 = tf.nn.dropout(hidden2_digit2, keep_prob)
        logits_digit2 = tf.matmul(hidden2_drop_digit2, weights_digit2_o) + biases_digit2_o
        
        hidden1_digit3 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_digit3_h1) + biases_digit3_h1)
        hidden1_drop_digit3 = tf.nn.dropout(hidden1_digit3, keep_prob)
        hidden2_digit3 = tf.nn.relu(tf.matmul(hidden1_drop_digit3, weights_digit3_h2) + biases_digit3_h2)
        hidden2_drop_digit3 = tf.nn.dropout(hidden2_digit3, keep_prob)
        logits_digit3 = tf.matmul(hidden2_drop_digit3, weights_digit3_o) + biases_digit3_o

        beta = 0.01
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_num_digits, tf_train_labels[:,0:3]) +
                              tf.nn.softmax_cross_entropy_with_logits(logits_digit1, tf_train_labels[:,3:13]) +
                              tf.nn.softmax_cross_entropy_with_logits(logits_digit2, tf_train_labels[:,13:23]) +
                              tf.nn.softmax_cross_entropy_with_logits(logits_digit3, tf_train_labels[:,23:33]) +
                              beta*tf.nn.l2_loss(weights_num_digits_h1) + 
                              beta*tf.nn.l2_loss(weights_num_digits_h2) + 
                              beta*tf.nn.l2_loss(weights_num_digits_o) + 
                              beta*tf.nn.l2_loss(weights_digit1_h1) + 
                              beta*tf.nn.l2_loss(weights_digit1_h2) + 
                              beta*tf.nn.l2_loss(weights_digit1_o) + 
                              beta*tf.nn.l2_loss(weights_digit2_h1) + 
                              beta*tf.nn.l2_loss(weights_digit2_h2) + 
                              beta*tf.nn.l2_loss(weights_digit2_o) +
                              beta*tf.nn.l2_loss(weights_digit3_h1) + 
                              beta*tf.nn.l2_loss(weights_digit3_h2) + 
                              beta*tf.nn.l2_loss(weights_digit3_o))
      
        # Optimizer.
        learning_rate = 0.001
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
      
        # Predictions for the training, validation, and test data.
        train_prediction_num_digits = tf.nn.softmax(logits_num_digits)
        train_prediction_digit1 = tf.nn.softmax(logits_digit1)
        train_prediction_digit2 = tf.nn.softmax(logits_digit2)
        train_prediction_digit3 = tf.nn.softmax(logits_digit3)

        #valid_prediction_num_digits = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights_num_digits) + biases_num_digits)
        valid_hidden1_num_digits = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_num_digits_h1) + biases_num_digits_h1)
        valid_hidden2_num_digits = tf.nn.relu(tf.matmul(valid_hidden1_num_digits, weights_num_digits_h2) + biases_num_digits_h2)
        valid_logits_num_digits = tf.matmul(valid_hidden2_num_digits, weights_num_digits_o) + biases_num_digits_o
        valid_prediction_num_digits = tf.nn.softmax(valid_logits_num_digits)
        #valid_prediction_digit1 = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights_digit1) + biases_digit1)
        valid_hidden1_digit1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_digit1_h1) + biases_digit1_h1)
        valid_hidden2_digit1 = tf.nn.relu(tf.matmul(valid_hidden1_digit1, weights_digit1_h2) + biases_digit1_h2)
        valid_logits_digit1 = tf.matmul(valid_hidden2_digit1, weights_digit1_o) + biases_digit1_o
        valid_prediction_digit1 = tf.nn.softmax(valid_logits_digit1)
        #valid_prediction_digit2 = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights_digit2) + biases_digit2)
        valid_hidden1_digit2 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_digit2_h1) + biases_digit2_h1)
        valid_hidden2_digit2 = tf.nn.relu(tf.matmul(valid_hidden1_digit2, weights_digit2_h2) + biases_digit2_h2)
        valid_logits_digit2 = tf.matmul(valid_hidden2_digit2, weights_digit2_o) + biases_digit2_o
        valid_prediction_digit2 = tf.nn.softmax(valid_logits_digit2)
        #valid_prediction_digit3 = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights_digit3) + biases_digit3)
        valid_hidden1_digit3 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_digit3_h1) + biases_digit3_h1)
        valid_hidden2_digit3 = tf.nn.relu(tf.matmul(valid_hidden1_digit3, weights_digit3_h2) + biases_digit3_h2)
        valid_logits_digit3 = tf.matmul(valid_hidden2_digit3, weights_digit3_o) + biases_digit3_o
        valid_prediction_digit3 = tf.nn.softmax(valid_logits_digit3)

        #test_prediction_num_digits = tf.nn.softmax(tf.matmul(tf_test_dataset, weights_num_digits) + biases_num_digits)
        test_hidden1_num_digits = tf.nn.relu(tf.matmul(tf_test_dataset, weights_num_digits_h1) + biases_num_digits_h1)
        test_hidden2_num_digits = tf.nn.relu(tf.matmul(test_hidden1_num_digits, weights_num_digits_h2) + biases_num_digits_h2)
        test_logits_num_digits = tf.matmul(test_hidden2_num_digits, weights_num_digits_o) + biases_num_digits_o
        test_prediction_num_digits = tf.nn.softmax(test_logits_num_digits)
        #test_prediction_digit1 = tf.nn.softmax(tf.matmul(tf_test_dataset, weights_digit1) + biases_digit1)
        test_hidden1_digit1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights_digit1_h1) + biases_digit1_h1)
        test_hidden2_digit1 = tf.nn.relu(tf.matmul(test_hidden1_digit1, weights_digit1_h2) + biases_digit1_h2)
        test_logits_digit1 = tf.matmul(test_hidden2_digit1, weights_digit1_o) + biases_digit1_o
        test_prediction_digit1 = tf.nn.softmax(test_logits_digit1)
        #test_prediction_digit2 = tf.nn.softmax(tf.matmul(tf_test_dataset, weights_digit2) + biases_digit2)
        test_hidden1_digit2 = tf.nn.relu(tf.matmul(tf_test_dataset, weights_digit2_h1) + biases_digit2_h1)
        test_hidden2_digit2 = tf.nn.relu(tf.matmul(test_hidden1_digit2, weights_digit2_h2) + biases_digit2_h2)
        test_logits_digit2 = tf.matmul(test_hidden2_digit2, weights_digit2_o) + biases_digit2_o
        test_prediction_digit2 = tf.nn.softmax(test_logits_digit2)
        #test_prediction_digit3 = tf.nn.softmax(tf.matmul(tf_test_dataset, weights_digit3) + biases_digit3)
        test_hidden1_digit3 = tf.nn.relu(tf.matmul(tf_test_dataset, weights_digit3_h1) + biases_digit3_h1)
        test_hidden2_digit3 = tf.nn.relu(tf.matmul(test_hidden1_digit3, weights_digit3_h2) + biases_digit3_h2)
        test_logits_digit3 = tf.matmul(test_hidden2_digit3, weights_digit3_o) + biases_digit3_o
        test_prediction_digit3 = tf.nn.softmax(test_logits_digit3)
              
        num_steps = 1000

        def accuracy(predictions, labels):
            ac_nd = 1.0 * np.sum(np.argmax(predictions[0], 1) == np.argmax(labels[:,0:3], 1)) / predictions[0].shape[0]
            ac_d1 = 1.0 * np.sum(np.argmax(predictions[1], 1) == np.argmax(labels[:,3:13], 1)) / predictions[1].shape[0]
            ac_d2 = 1.0 * np.sum(np.argmax(predictions[2], 1) == np.argmax(labels[:,13:23], 1)) / predictions[2].shape[0]
            ac_d3 = 1.0 * np.sum(np.argmax(predictions[3], 1) == np.argmax(labels[:,23:33], 1)) / predictions[3].shape[0]
            #ac_d4 = 1.0 * np.sum(np.argmax(predictions[4], 1) == np.argmax(labels[:,35:45], 1)) / predictions[4].shape[0]
            #ac_d5 = 1.0 * np.sum(np.argmax(predictions[5], 1) == np.argmax(labels[:,45:55], 1)) / predictions[5].shape[0]
            overall = ac_nd * ac_d1 * ac_d2 * ac_d3 
            return ac_nd, ac_d1, ac_d2, ac_d3, overall
            
        start = time.time()
        with tf.Session(graph=graph) as session:
            # This is a one-time operation which ensures the parameters get initialized as
            # we described in the graph: random weights for the matrix, zeros for the biases. 
             tf.initialize_all_variables().run()
             print('Initialized')
             for step in range(num_steps):
                 # Pick an offset within the training data, which has been randomized.
                 # Note: we could use better randomization across epochs.
                 offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                 # Generate a minibatch.
                 batch_data = train_dataset[offset:(offset + batch_size), :]
                 batch_labels = train_labels[offset:(offset + batch_size), :]
                 # Prepare a dictionary telling the session where to feed the minibatch.
                 # The key of the dictionary is the placeholder node of the graph to be fed,
                 # and the value is the numpy array to feed to it.
                 feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: 0.5}
                 # Run the computations. We tell .run() that we want to run the optimizer,
                 # and get the loss value and the training predictions returned as numpy
                 # arrays.
                 _, l, prediction_num_digits, prediction_digit1, prediction_digit2, prediction_digit3 = session.run([optimizer, loss, train_prediction_num_digits, 
                                                  train_prediction_digit1, train_prediction_digit2, 
                                                  train_prediction_digit3], feed_dict=feed_dict)
                 if (step % 100 == 0):
                     print("Minibatch Loss at step {}: {}".format(step, l))
                     print('Minibatch Training accuracy: {}'.format(accuracy(
                            (prediction_num_digits, prediction_digit1, prediction_digit2, 
                            prediction_digit3), batch_labels)))
                     # Calling .eval() on valid_prediction is basically like calling run(), but
                     # just to get that one numpy array. Note that it recomputes all its graph
                     # dependencies.
                     print('Validation accuracy: {}'.format(accuracy(
                            (valid_prediction_num_digits.eval(feed_dict={tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:1.0}),
                            valid_prediction_digit1.eval(feed_dict={tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:1.0}), 
                            valid_prediction_digit2.eval(feed_dict={tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:1.0}), 
                            valid_prediction_digit3.eval(feed_dict={tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:1.0})), 
                            valid_labels)))
             print('Test accuracy: {}'.format(accuracy((
                    test_prediction_num_digits.eval(feed_dict={keep_prob:1.0}), 
                    test_prediction_digit1.eval(feed_dict={keep_prob:1.0}), 
                    test_prediction_digit2.eval(feed_dict={keep_prob:1.0}), 
                    test_prediction_digit3.eval(feed_dict={keep_prob:1.0})),
                    test_labels)))
             end = time.time()
             print("Time taken to train database : {} seconds".format(end - start))