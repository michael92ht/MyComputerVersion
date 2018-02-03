# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import random_seed

PIXEL_DEPTH = 255
IMAGE_SIZE = 64

DATASET_PATH = "C:\codes\earthquake\datasets\waveform"
TRAIN_FILE = os.path.join(DATASET_PATH, 'train_datas.txt')
TEST_FILE = os.path.join(DATASET_PATH, 'test_datas.txt')


PICKLED_PATH = "C:\codes\earthquake\datasets\pickled"
train_waveform_pickled = os.path.join(PICKLED_PATH, 'train_waveform_pickled')
train_labels_pickled = os.path.join(PICKLED_PATH, 'train_labels_pickled')
test_waveform_pickled = os.path.join(PICKLED_PATH, 'test_waveform_pickled')
test_labels_pickled = os.path.join(PICKLED_PATH, 'test_labels_pickled')


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)

        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)

        # assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        images = images.astype(np.float32)
        # images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# 读取数据集文件，返回特征及标签列表
def read_data_file(data_path):
    with open(data_path, "r") as f:
        lines = f.readlines()
        count = len(lines)
        features, labels = [], []
        for line in lines:
            content = line.split(":")
            label = int(content[0])
            data = np.array([float(x) for x in content[1].split(",")])
            feature = data.reshape((IMAGE_SIZE, IMAGE_SIZE, 1))

            assert label in [0, 1]
            features.append(feature)
            labels.append(label)

    features = np.array(features)
    labels = np.array(labels)
    labels = dense_to_one_hot(labels, 2)
    assert features.shape == (count, IMAGE_SIZE, IMAGE_SIZE, 1)
    assert labels.shape == (count, 2)

    return features, labels


def pickle_dataset(overflow=False):

    train_waveforms, train_labels = read_data_file(TRAIN_FILE)
    test_waveforms, test_labels = read_data_file(TEST_FILE)

    # 将图像数据集序列化
    print("Dump train waveform.")
    if overflow:
        half = int(len(train_waveforms) / 2)
        train_waveforms[: half].dump(train_waveform_pickled + "_0")
        train_waveforms[half:].dump(train_waveform_pickled + "_1")
    else:
        train_waveforms.dump(train_waveform_pickled)
    train_labels.dump(train_labels_pickled)
    test_waveforms.dump(test_waveform_pickled)
    test_labels.dump(test_labels_pickled)


def read_data_sets(validation_size=2000, overflow=False):
    print("Begin read datasets.")
    if overflow:
        train_waveforms = np.load(train_waveform_pickled)
    else:
        left = np.load(train_waveform_pickled + "_0")
        right = np.load(train_waveform_pickled + "_1")
        train_waveforms = np.concatenate((left, right))
    train_labels = np.load(train_labels_pickled)
    test_waveforms = np.load(test_waveform_pickled)
    test_labels = np.load(test_labels_pickled)

    if not 0 <= validation_size <= len(train_waveforms):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_waveforms), validation_size))

    validation_waveforms = train_waveforms[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_waveforms = train_waveforms[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_waveforms, train_labels)
    validation = DataSet(validation_waveforms, validation_labels)
    test = DataSet(test_waveforms, test_labels)
    print("Read all datasets done!")

    return base.Datasets(train=train, validation=validation, test=test)


def load_dataset(overflow=False):
    return read_data_sets(overflow)


if __name__ == "__main__":
    # data = "wg"
    # pickle_path = "images"
    pickle_dataset(overflow=True)
    # wg = load_dataset(overflow=True)
    # for i in range(10):
    #     x, y = wg.validation.next_batch(64)
    #     print(x.shape, y.shape)
    #     print(y, y[0])
        # h = imgs[0][0].reshape(64, 64)
        # print(imgs[1][0])
        # img = Image.fromarray(np.uint8(h * 255), mode='L')
        # img.save(str(i) + '_temp.jpg')

