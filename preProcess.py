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
from operator import add
import cv2

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  if force or not os.path.exists(filename):
    print('Attempting to download:', filename) 
    filename, _ = urlretrieve(url + filename, filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s directory already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
        
def maybe_pickle(filename, train_dataset,
                            train_labels,
                            valid_dataset,
                            valid_labels,
                            test_dataset,
                            test_labels, force=False):
    if os.path.exists(filename) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping pickling.' % filename)
    else:
        print('Pickling %s.' % filename)

        try:
            f = open(filename, 'wb')
            save = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
                }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', filename, ':', e)
            raise

        statinfo = os.stat(filename)
        print('Compressed pickle size:', statinfo.st_size)

def getH5ArrayValues(h5File, h5array):
    arrVal = []
    if len(h5array) > 1:
        for i in range(0, len(h5array)):
            nextWidth = h5File[h5array[i][0]]
            arrVal.append(int(nextWidth[0][0]))
    else:
        arrVal.append(int(h5array[0][0]))
    return arrVal

def getH5LabelValues(h5File, h5array):
    digArr = [0, 0, 0, 0, 0]
    num_digits = len(h5array)
    if len(h5array) > 1:
        for i in range(0, num_digits if num_digits < 6 else 5):
            nextWidth = h5File[h5array[i][0]]
            digArr[i] = int(nextWidth[0][0])
    else:
        digArr[0] = int(h5array[0][0])
    return num_digits, digArr

def getShrinkWrap(name, leftArr, topArr, widthArr, heightArr):
    img = cv2.imread(name)
    imgH, imgW, channels = img.shape
    if debug:
        print("  Image size: {} x {}, {} channels".format(imgH, imgW, channels))
    left = min(leftArr)
    top = min(topArr)
    right = max(map(add, leftArr, widthArr))
    bottom = max(map(add, topArr, heightArr))
    # Expand by 30% in x,y
    xExtent = right - left
    yExtent = bottom - top
    xAdd = int(xExtent * 0.15)
    yAdd = int(yExtent * 0.15)
    newX1 = left - xAdd if left - xAdd > 0 else 0
    newY1 = top - yAdd if top - yAdd > 0 else 0
    newX2 = right + xAdd if right + xAdd < imgW else imgW
    newY2 = bottom + yAdd if bottom + yAdd < imgH else imgH
    bBox = [newX1, newY1, newX2, newY2]
    return img, bBox

def cropImage(img, bBox):
    if debug:
        cv2.imshow("Orig", img)
        cv2.waitKey()
    cropImg = img[bBox[1]:bBox[3], bBox[0]:bBox[2]]
    if debug:
        cv2.imshow("Cropped", cropImg)
        cv2.waitKey(0)
    return cropImg

def resizeImage(image, sizeX, sizeY):
    resizedImg = cv2.resize(image, (sizeX, sizeY))
    if debug:
        cv2.imshow("Resized", resizedImg)
        cv2.waitKey(0)
    return resizedImg

def readMD5Mat(dname, fname, num_images, image_sizeX, image_sizeY, channels):
    # Need to use h5py to read these mat files as they are v7.3
    filename = './' + dname + '/' + fname
    print("Reading images and labels from {}. This may take some time.".format(filename))
    h5pyFile = h5py.File(filename)
    digitStruct = h5pyFile.get('digitStruct')
    bboxes = digitStruct['bbox']
    names = digitStruct['name']
    dataset = np.ndarray(shape=(num_images, image_sizeX, image_sizeY, channels), dtype=np.float32)
    labelset = np.ndarray(shape=(num_images, 6), dtype=np.uint8)

    #for i in range(1, 2):
    for i in range(0, num_images):
        #print(i)
        imgNameArr = h5pyFile[names[i][0]]
        imgName = ""
        for j in range(0, len(imgNameArr)):
            nextChar = str(unichr(imgNameArr[[j][0]]))
            imgName += nextChar
        imgName = './' + dname + '/' + imgName
        if debug:
            print("Filename: {}".format(imgName))
        bBoxArr = h5pyFile[bboxes[i][0]]
        leftArr = bBoxArr['left']
        topArr = bBoxArr['top']
        heightArr = bBoxArr['height']
        widthArr = bBoxArr['width']
        labelArr = bBoxArr['label']
        num_labels, labels = getH5LabelValues(h5pyFile, labelArr)
        if num_labels > 5:
            print("** Image {} has more than 5 digits! Found {}".format(imgName, num_labels))
        if num_labels <= 0:
            print("** Image {} has less than 0 digits! Found {}".format(imgName, num_labels))
        if debug:
            print("  Num_Labels: {}".format(num_labels))
            print("  Labels: {}".format(labels))
        lefts = getH5ArrayValues(h5pyFile, leftArr)
        if debug:
            print("  Lefts: {}".format(lefts))
        tops = getH5ArrayValues(h5pyFile, topArr)
        if debug:
            print("  Tops: {}".format(tops))
        heights = getH5ArrayValues(h5pyFile, heightArr)
        if debug:
            print("  Heights: {}".format(heights))
        widths = getH5ArrayValues(h5pyFile, widthArr)
        if debug:
            print("  Widths: {}".format(widths))
        img, imgROI = getShrinkWrap(imgName, lefts, tops, widths, heights)
        if debug:
            print("  imgROI: {}".format(imgROI))
        croppedImg = cropImage(img, imgROI)
        resizedImg = resizeImage(croppedImg, image_sizeX, image_sizeY)
        # Normalize the dataset here also
        dataset[i] = resizedImg / 256.0
        labelset[i] = ([num_labels] + labels)
    return dataset, labelset

# START OF MAIN PROGRAM
debug = 0
num_classes = 10
np.random.seed(133)
image_sizeX = 32  # Pixel width
image_sizeY = 32  # Pixel height
num_train_images = 33402
num_test_images = 13068
num_train_items = 23402
channels = 3
pixel_depth = 255.0  # Number of levels per pixel.
pickleFilename = "./svhn.pickle"
force = 0

url = 'http://ufldl.stanford.edu/housenumbers/'
last_percent_reported = None

train_filename = maybe_download('train.tar.gz', 404141560)
test_filename = maybe_download('test.tar.gz', 276555967)

# Uncompress the tar files if needed
maybe_extract(train_filename)
maybe_extract(test_filename)

# Read in the training and test datasets and labels from the MD5 .mat file and .png images
# As part of this, I resize the images to a standard size
train_all_dataset,  train_all_labels = readMD5Mat('train', 'digitStruct.mat', num_train_images, image_sizeX, image_sizeY, channels)
test_dataset, test_labels = readMD5Mat('test', 'digitStruct.mat', num_test_images, image_sizeX, image_sizeY, channels)

# Data is already randomized, so no need for further randomization

# Split train_dataset and labels into a training set and a CV set
train_dataset = train_all_dataset[:num_train_items]
train_labels = train_all_labels[:num_train_items]
valid_dataset = train_all_dataset[num_train_items:]
valid_labels = train_all_labels[num_train_items:]

# Lets see how many of each sequence of digits there are
numNums = [item[0] for item in train_labels]
for i in range(0,7):
    print("Number of train items with {} digit(s): {}".format(i, numNums.count(i)))
numNums = [item[0] for item in valid_labels]
for i in range(0,7):
    print("Number of validation items with {} digit(s): {}".format(i, numNums.count(i)))
numNums = [item[0] for item in test_labels]
for i in range(0,7):
    print("Number of test items with {} digit(s): {}".format(i, numNums.count(i)))

# Lets store these for future use as pickle files
# train_datasets = maybe_pickle(data_folders, min_num_images_per_class, force)
maybe_pickle(pickleFilename, 
            train_dataset,
            train_labels,
            valid_dataset,
            valid_labels,
            test_dataset,
            test_labels,
            force)

print("Data Pre-processing Complete!")
