#!/usr/bin/python

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile

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


pickle_file = './svhn_3digits_gray.pickle'
use_cnn = True
image_sizeX =32
image_sizeY = 32
num_channels = 1
num_digits = 3
num_labels = 11

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

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

graph = tf.Graph()

tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_sizeX, image_sizeY, num_channels))
tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_digits + num_digits * num_labels))
tf_valid_dataset = tf.constant(valid_dataset)
tf_test_dataset = tf.constant(test_dataset)

model_name = "./logs/svhm_cnn_dep_16_ps_5_reg_0.01_lr_0.002_nnl1_1024_nnl2_512_bs_128_ts_full_06.01PM_November_03_2016_"


with tf.Session(graph=graph) as session:
  print("Load graph...")
  with gfile.FastGFile(model_name + ".proto", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
  print("Map variables...")
  saver = tf.train.Saver()
  saver.restore(session, "./logs/svhm_cnn_dep_16_ps_5_reg_0.01_lr_0.002_nnl1_1024_nnl2_512_bs_128_ts_full_06.01PM_November_03_2016_")

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
