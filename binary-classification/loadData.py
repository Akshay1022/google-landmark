import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def read_images(trainingpath, image_size, classes):
    images = []
    labels = []
    image_names = []
    cls = []

    print('Will start reading training images')
    for each_class in classes:   
        index = classes.index(each_class)
        print('Now going to read {} files (Index: {})'.format(each_class, index))
        path = os.path.join(trainingpath, each_class, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            image_names.append(flbase)
            cls.append(each_class)
    images = np.array(images)
    labels = np.array(labels)
    image_names = np.array(image_names)
    cls = np.array(cls)

    return images, labels, image_names, cls


class DataSet(object):

  def __init__(self, images, labels, image_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._image_names = image_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def image_names(self):
    return self._image_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._image_names[start:end], self._cls[start:end]


def read_train_data(trainingpath, image_size, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, image_names, cls = read_images(trainingpath, image_size, classes)
  images, labels, image_names, cls = shuffle(images, labels, image_names, cls)  

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_image_names = image_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_image_names = image_names[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_image_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_image_names, validation_cls)

  return data_sets