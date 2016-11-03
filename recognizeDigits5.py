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
import datetime

def reformat(dataset, labels):
  if use_cnn:
      dataset = dataset.reshape((-1, image_sizeX, image_sizeY, num_channels)).astype(np.float32)
  else:
      dataset = dataset.reshape((-1, image_sizeX * image_sizeY * num_channels)).astype(np.float32)
  digit_number = labels[:,0]
  num_digits_encoded = (np.arange(num_digits) == digit_number[:,None]-1).astype(np.float32)
  digit1 = labels[:,1]
  digit1_encoded = (np.arange(num_labels) == digit1[:,None]).astype(np.float32)
  digit2 = labels[:,2]
  digit2_encoded = (np.arange(num_labels) == digit2[:,None]).astype(np.float32)
  digit3 = labels[:,3]
  digit3_encoded = (np.arange(num_labels) == digit3[:,None]).astype(np.float32)

  labels = np.hstack((num_digits_encoded, digit1_encoded, digit2_encoded, digit3_encoded))
  return dataset, labels

def model_name():
  model_name = "svhm_" 
  if use_cnn:
    model_name += "cnn_"
    model_name += "dep_" + str(depth) + "_"
    model_name += "ps_" + str(patch_size) + "_"
  if use_regularization:
    model_name += "reg_" + str(reg_beta) + "_"
  if use_learning_rate_decay:
    model_name += "lrd_"
  model_name += "lr_" + str(initial_learning_rate) + "_"
  if use_dropout:
    model_name += "do_" + "kp_" + str(dropout_keep_prob) + "_"
  model_name += "nnl1_" + str(hidden_layer1_size) + "_" 
  model_name += "nnl2_" + str(hidden_layer2_size) + "_" 
  model_name += "bs_" + str(batch_size) +"_"
  model_name += "ts_"
  if train_subset == -1:
    model_name += "full_"
  else:
    model_name += str(train_subset) + "_"
  model_name += datetime.datetime.now().strftime("%I.%M%p_%B_%d_%Y_")
  return model_name
            
# START OF MAIN PROGRAM
debug = 0
image_sizeX = 32
image_sizeY = 32
num_channels = 1 # grayscale
num_digits = 3
num_labels = 11  #  0 = 'no digit'. 1..10 = digits 1 to 10(0)
epochs = 3500
digit1_index = num_digits
digit2_index = num_digits + num_labels
digit3_index = num_digits + (num_labels * 2)

batch_size = 128
patch_size = 5
depth = 16
hidden_layer1_size =  1024
hidden_layer2_size =  512
dropout_keep_prob = 0.5
use_cnn = True
use_regularization = True
reg_beta = 0.01
use_learning_rate_decay = False
use_dropout = False
initial_learning_rate = 0.002
#train_subset = 130 # Use -1 for full dataset
train_subset = -1 # Use -1 for full dataset
compute_validation = False
compute_test = True
compute_single_inference=True

# Create log directory if needed
dir = "./logs"
try:
  os.stat(dir)
except:
  os.mkdir(dir) 
# Create models directory if needed
dir = "./models"
try:
  os.stat(dir)
except:
  os.mkdir(dir) 

model_file_name = "./models/" + model_name()
log_file_name = "./logs/" + model_name() + ".log"
print("Writing to log file {}".format(log_file_name))
log_file = open(log_file_name, 'w')

pickle_file = './svhn_3digits_gray.pickle'

# Start by reading in the pickle datasets
print("Reading pickle file {}".format(pickle_file), file=log_file)
with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  
print('Training set', train_dataset.shape, train_labels.shape, file=log_file)
print('Validation set', valid_dataset.shape, valid_labels.shape, file=log_file)
print('Test set', test_dataset.shape, test_labels.shape, file=log_file)

if debug:
    # Let's draw one of the images to ensure they are read correctly
    img_index = 1000
    plt.imshow(train_dataset[img_index,:,:,:])
    print (train_labels[img_index])
    plt.show()

# Resize dataset to subset if needed
if train_subset == -1:
    train_subset = len(train_dataset)
train_dataset = train_dataset[:train_subset, :]
train_labels = train_labels[:train_subset, :]

if debug:
    print("Number of training samples : {}". format(train_subset), file=log_file)
        
# Reformat to encode labels etc.
# labels is not of the format (one_shot_enc_num_digits, one_shot_enc_digit1, ... , one_shot_enc_digit5)
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape, file=log_file)
print('Validation set', valid_dataset.shape, valid_labels.shape, file=log_file)
print('Test set', test_dataset.shape, test_labels.shape, file=log_file)
    
graph = tf.Graph()
with graph.as_default():
    
    # Input data.
    if use_cnn:
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_sizeX, image_sizeY, num_channels))
    else:
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_sizeX * image_sizeY * num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_digits + num_digits * num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
      
    # Variables.
    if use_cnn:
        cnn_layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth]))
        cnn_layer1_weights = tf.get_variable("cnn_layer1_weights", 
                                            shape = [patch_size, patch_size, num_channels, depth],
                                            initializer=tf.contrib.layers.xavier_initializer())
        cnn_layer1a_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth]))
        cnn_layer1a_weights = tf.get_variable("cnn_layer1a_weights", 
                                            shape = [patch_size, patch_size, depth, depth],
                                            initializer=tf.contrib.layers.xavier_initializer())
        cnn_layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth]))
        cnn_layer2_weights = tf.get_variable("cnn_layer2_weights", 
                                        shape = [patch_size, patch_size, depth, depth],
                                        initializer=tf.contrib.layers.xavier_initializer())
        #cnn_layer2a_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth]))
        #cnn_layer2a_weights = tf.get_variable("cnn_layer2a_weights", 
        #                                shape = [patch_size, patch_size, depth, depth],
        #                                initializer=tf.contrib.layers.xavier_initializer())
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

    if use_cnn:
        cnn_layer1_biases = tf.Variable(tf.zeros([depth]))
        cnn_layer1a_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
        cnn_layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
        #cnn_layer2a_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    biases_num_digits_h1 = tf.Variable(tf.constant(1.0, shape=[hidden_layer1_size]))
    biases_num_digits_h2 = tf.Variable(tf.constant(1.0, shape=[hidden_layer2_size]))
    biases_num_digits_o = tf.Variable(tf.constant(1.0, shape=[num_digits]))
    biases_digit1_h1 = tf.Variable(tf.constant(1.0, shape=[hidden_layer1_size]))
    biases_digit1_h2 = tf.Variable(tf.constant(1.0, shape=[hidden_layer2_size]))
    biases_digit1_o = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    biases_digit2_h1 = tf.Variable(tf.constant(1.0, shape=[hidden_layer1_size]))
    biases_digit2_h2 = tf.Variable(tf.constant(1.0, shape=[hidden_layer2_size]))
    biases_digit2_o = tf.Variable(tf.constant(1.0, shape=[num_labels]))
    biases_digit3_h1 = tf.Variable(tf.constant(1.0, shape=[hidden_layer1_size]))
    biases_digit3_h2 = tf.Variable(tf.constant(1.0, shape=[hidden_layer2_size]))
    biases_digit3_o = tf.Variable(tf.constant(1.0, shape=[num_labels]))
        
    keep_prob = tf.placeholder(tf.float32)

    # Training computation.
    if use_cnn:
        def cnn_model(data):
            conv1 = tf.nn.conv2d(data, cnn_layer1_weights, [1, 1, 1, 1], padding='SAME')
            hidden1 = tf.nn.relu(conv1 + cnn_layer1_biases)
            conv1a = tf.nn.conv2d(hidden1, cnn_layer1a_weights, [1, 1, 1, 1], padding='SAME')
            hidden1a = tf.nn.relu(conv1a + cnn_layer1a_biases)
            pooling1 = tf.nn.max_pool(hidden1a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            conv2 = tf.nn.conv2d(pooling1, cnn_layer2_weights, [1, 1, 1, 1], padding='SAME')
            hidden2 = tf.nn.relu(conv2 + cnn_layer2_biases)
            #conv2a = tf.nn.conv2d(hidden2, cnn_layer2a_weights, [1, 1, 1, 1], padding='SAME')
            #hidden2a = tf.nn.relu(conv2a + cnn_layer2a_biases)
            pooling2 = tf.nn.max_pool(hidden2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            shape = pooling2.get_shape().as_list()
            return tf.reshape(pooling2, [shape[0], shape[1] * shape[2] * shape[3]])
            
    def model_num_digits(data):
        hidden1_num_digits = tf.nn.relu(tf.matmul(data, weights_num_digits_h1) + biases_num_digits_h1)
        if use_dropout:
            hidden1_drop_num_digits = tf.nn.dropout(hidden1_num_digits, keep_prob)
        else:
            hidden1_drop_num_digits = hidden1_num_digits
        hidden2_num_digits = tf.nn.relu(tf.matmul(hidden1_drop_num_digits, weights_num_digits_h2) + biases_num_digits_h2)
        if use_dropout:
            hidden2_drop_num_digits = tf.nn.dropout(hidden2_num_digits, keep_prob)
        else:
            hidden2_drop_num_digits = hidden2_num_digits
        return tf.matmul(hidden2_drop_num_digits, weights_num_digits_o) + biases_num_digits_o
        
    def model_digit1(data):
        hidden1_digit1 = tf.nn.relu(tf.matmul(data, weights_digit1_h1) + biases_digit1_h1)
        if use_dropout:
            hidden1_drop_digit1 = tf.nn.dropout(hidden1_digit1, keep_prob)
        else:
            hidden1_drop_digit1 = hidden1_digit1
        hidden2_digit1 = tf.nn.relu(tf.matmul(hidden1_drop_digit1, weights_digit1_h2) + biases_digit1_h2)
        if use_dropout:
            hidden2_drop_digit1 = tf.nn.dropout(hidden2_digit1, keep_prob)
        else:
            hidden2_drop_digit1 = hidden2_digit1
        return tf.matmul(hidden2_drop_digit1, weights_digit1_o) + biases_digit1_o
        
    def model_digit2(data):    
        hidden1_digit2 = tf.nn.relu(tf.matmul(data, weights_digit2_h1) + biases_digit2_h1)
        if use_dropout:
            hidden1_drop_digit2 = tf.nn.dropout(hidden1_digit2, keep_prob)
        else:
            hidden1_drop_digit2 = hidden1_digit2
        hidden2_digit2 = tf.nn.relu(tf.matmul(hidden1_drop_digit2, weights_digit2_h2) + biases_digit2_h2)
        if use_dropout:
            hidden2_drop_digit2 = tf.nn.dropout(hidden2_digit2, keep_prob)
        else:
            hidden2_drop_digit2 = hidden2_digit2
        return tf.matmul(hidden2_drop_digit2, weights_digit2_o) + biases_digit2_o
        
    def model_digit3(data):    
        hidden1_digit3 = tf.nn.relu(tf.matmul(data, weights_digit3_h1) + biases_digit3_h1)
        if use_dropout:
            hidden1_drop_digit3 = tf.nn.dropout(hidden1_digit3, keep_prob)
        else:
            hidden1_drop_digit3 = hidden1_digit3
        hidden2_digit3 = tf.nn.relu(tf.matmul(hidden1_drop_digit3, weights_digit3_h2) + biases_digit3_h2)
        if use_dropout:
            hidden2_drop_digit3 = tf.nn.dropout(hidden2_digit3, keep_prob)
        else:
            hidden2_drop_digit3 = hidden2_digit3
        return tf.matmul(hidden2_drop_digit3, weights_digit3_o) + biases_digit3_o
        
    def full_model(data):
        if use_cnn:    
            cnn_data = cnn_model(data)
        else:
            cnn_data = tf_train_dataset
        lnd = model_num_digits(cnn_data)
        ld1 = model_digit1(cnn_data)
        ld2 = model_digit2(cnn_data)
        ld3 = model_digit3(cnn_data)
        return lnd, ld1, ld2, ld3
        
    # Instanciate the model
    logits_num_digits, logits_digit1, logits_digit2, logits_digit3 = full_model(tf_train_dataset)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits_num_digits, tf_train_labels[:,0:3]) + \
        tf.nn.softmax_cross_entropy_with_logits(logits_digit1, tf_train_labels[:,digit1_index:digit2_index]) + \
        tf.nn.softmax_cross_entropy_with_logits(logits_digit2, tf_train_labels[:,digit2_index:digit3_index]) + \
        tf.nn.softmax_cross_entropy_with_logits(logits_digit3, tf_train_labels[:,digit3_index:digit3_index + num_labels])
    if use_cnn and use_regularization:
        loss += reg_beta*tf.nn.l2_loss(cnn_layer1_weights) + \
		reg_beta*tf.nn.l2_loss(cnn_layer1a_weights) + \
                reg_beta*tf.nn.l2_loss(cnn_layer2_weights)
    if use_regularization:
        loss += reg_beta*tf.nn.l2_loss(weights_num_digits_h1) + \
                reg_beta*tf.nn.l2_loss(weights_num_digits_h2) + \
                reg_beta*tf.nn.l2_loss(weights_num_digits_o) + \
                reg_beta*tf.nn.l2_loss(weights_digit1_h1) + \
                reg_beta*tf.nn.l2_loss(weights_digit1_h2) + \
                reg_beta*tf.nn.l2_loss(weights_digit1_o) + \
                reg_beta*tf.nn.l2_loss(weights_digit2_h1) + \
                reg_beta*tf.nn.l2_loss(weights_digit2_h2) + \
                reg_beta*tf.nn.l2_loss(weights_digit2_o) + \
                reg_beta*tf.nn.l2_loss(weights_digit3_h1) + \
                reg_beta*tf.nn.l2_loss(weights_digit3_h2) + \
                reg_beta*tf.nn.l2_loss(weights_digit3_o)
    loss = tf.reduce_mean(loss)
      
    # Optimizer.
    global_step = tf.Variable(0) # count the number of steps taken.
    if use_learning_rate_decay:
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, 3500, 0.96, staircase=True)
    else:
        learning_rate = initial_learning_rate
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # Don't ever decay learning with Adam
    optimizer = tf.train.AdamOptimizer(initial_learning_rate).minimize(loss, global_step=global_step)
      
    # Predictions for the training, validation, and test data.
    train_prediction_num_digits = tf.nn.softmax(logits_num_digits)
    train_prediction_digit1 = tf.nn.softmax(logits_digit1)
    train_prediction_digit2 = tf.nn.softmax(logits_digit2)
    train_prediction_digit3 = tf.nn.softmax(logits_digit3)
        
    valid_prediction_num_digits, valid_prediction_digit1, valid_prediction_digit2, valid_prediction_digit3 = full_model(tf_valid_dataset)       
    test_prediction_num_digits, test_prediction_digit1, test_prediction_digit2, test_prediction_digit3 = full_model(tf_test_dataset)
              
    def accuracy(predictions, labels):
        ac_nd = 1.0 * np.sum(np.argmax(predictions[0], 1) == np.argmax(labels[:,0:num_digits], 1)) / predictions[0].shape[0]
        ac_d1 = 1.0 * np.sum(np.argmax(predictions[1], 1) == np.argmax(labels[:,digit1_index:digit2_index], 1)) / predictions[1].shape[0]
        ac_d2 = 1.0 * np.sum(np.argmax(predictions[2], 1) == np.argmax(labels[:,digit2_index:digit3_index], 1)) / predictions[2].shape[0]
        ac_d3 = 1.0 * np.sum(np.argmax(predictions[3], 1) == np.argmax(labels[:,digit3_index:digit3_index + num_labels], 1)) / predictions[3].shape[0]
        #overall = ac_nd * ac_d1 * ac_d2 * ac_d3 
        overall = (ac_nd + ac_d1 + ac_d2 + ac_d3) / 4.0
        return ac_nd, ac_d1, ac_d2, ac_d3, overall
        
    # Add variables for viewing in Tensorboard
    for value in [loss]:
        tf.scalar_summary(value.op.name, value)           
    summaries = tf.merge_all_summaries()

    # Get ready to save the model          
    saver = tf.train.Saver()

    # Train the model        
    start = time.time()
    with tf.Session(graph=graph) as session:
        summary_writer = tf.train.SummaryWriter('log_svhn_graph', session.graph)
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the biases. 
        tf.initialize_all_variables().run()
        print('Initialized', file=log_file)
        for step in range(epochs):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob: dropout_keep_prob}
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            summary_str, _, l, prediction_num_digits, prediction_digit1, prediction_digit2, prediction_digit3 = session.run([summaries, optimizer, loss, train_prediction_num_digits, 
                                            train_prediction_digit1, train_prediction_digit2, 
                                            train_prediction_digit3], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            if (step % 100 == 0 or step == epochs-1):
                print("Minibatch Loss at step {} of {}: {}".format(step, epochs, l), file=log_file)
                print("Minibatch Loss at step {} of {}: {}".format(step, epochs, l))
                print('Minibatch Training accuracy: {}'.format(accuracy(
                    (prediction_num_digits, prediction_digit1, prediction_digit2, 
                    prediction_digit3), batch_labels)), file=log_file)
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.
		log_file.flush()
                if compute_validation:
                    print('Validation accuracy: {}'.format(accuracy(
                        (valid_prediction_num_digits.eval(feed_dict={tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:1.0}),
                        valid_prediction_digit1.eval(feed_dict={tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:1.0}), 
                        valid_prediction_digit2.eval(feed_dict={tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:1.0}), 
                        valid_prediction_digit3.eval(feed_dict={tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:1.0})), 
                        valid_labels)), file=log_file)
        if compute_test:
	    test_acc = accuracy((
                test_prediction_num_digits.eval(feed_dict={keep_prob:1.0}), 
                test_prediction_digit1.eval(feed_dict={keep_prob:1.0}), 
                test_prediction_digit2.eval(feed_dict={keep_prob:1.0}), 
                test_prediction_digit3.eval(feed_dict={keep_prob:1.0})),
                test_labels)
            print('Test accuracy: {}'.format(test_acc), file=log_file)
            print('Test accuracy: {}'.format(test_acc))
        end = time.time()
        print("Time taken to train database : {} seconds".format(end - start), file=log_file)
        print("Time taken to train database : {} seconds".format(end - start))
	log_file.close()

        # Save the model
        print("Saving model to {}".format(model_file_name))
        save_path = saver.save(session, model_file_name)
             
        # let's see what get predicted for a single image
	if compute_single_inference:
            start = time.time()
            input_image = test_dataset[0]
            input_label = test_labels[0]
            if use_cnn:
                input_placeholder = tf.placeholder(tf.float32, shape=(1, image_sizeX, image_sizeY, num_channels))
            else:
                input_placeholder = tf.placeholder(tf.float32, shape=(1, image_sizeX * image_sizeY * num_channels))
            label_placeholder = tf.placeholder(tf.float32, shape=(1, num_digits + num_digits * num_labels))
            input_image = input_image[np.newaxis, ...]
            input_label = input_label[np.newaxis, ...]
            feed_dict = {input_placeholder : input_image, label_placeholder : input_label, keep_prob: 1.0}
            test_pnd, test_pd1, test_pd2, test_pd3 = full_model(input_placeholder)
            pnd, pd1, pd2, pd3 = session.run([test_pnd, test_pd1, test_pd2, test_pd3], feed_dict=feed_dict)
            end = time.time()
            print("Time taken for single inference : {} seconds".format(end - start))
            print("Test image labels: {}".format(test_labels[0]))
            print("Neural Network predicted : {}".format([pnd, pd1, pd2, pd3]))
            print("Predicted num digits : {}".format(np.argmax(pnd)+1))
            print("Predicted number (0=noNum, 1...10= 1 to 0) : {}{}{}".format(np.argmax(pd1), np.argmax(pd2), np.argmax(pd3)))
            plt.imshow(input_image.reshape(32, 32), cmap=plt.cm.binary)
            plt.show()
