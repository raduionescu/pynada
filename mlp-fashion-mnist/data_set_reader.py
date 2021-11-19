from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import pdb
import cv2 as cv

import mnist_reader


class DataSetReader:

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        self.num_samples = len(labels)
        self.end_index = 0

    def next_batch(self, bach_size=32):
        if self.end_index == self.num_samples:
            self.end_index = 0
            self.dataset, self.labels = shuffle(self.dataset, self.labels)

        start_index = self.end_index
        self.end_index += bach_size
        self.end_index = min(self.end_index, self.num_samples)

        return self.dataset[start_index:self.end_index], self.labels[start_index:self.end_index]


def process_images(images):
    images = images.reshape((-1, 28, 28, 1))
    images = np.pad(images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    return np.array(images / 255.0, np.float32)


def one_hot_encoding(labels, num_classes=10):
    encoding = np.zeros((len(labels), num_classes))
    for idx, label in enumerate(labels):
        encoding[idx, label] = 1

    return encoding


def create_readers():
    X_train, y_train = mnist_reader.load_mnist('../fashion-mnist', kind='train')
    X_test, y_test = mnist_reader.load_mnist('../fashion-mnist', kind='t10k')
    X_train = np.float32(X_train) / 255.0
    X_test = np.float32(X_test) / 255.0
    y_train = one_hot_encoding(y_train)
    y_test = one_hot_encoding(y_test)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=10000, random_state=357) 
    
    mean_pixel = np.mean(X_train)
    X_train -= mean_pixel
    X_val -= mean_pixel
    X_test -= mean_pixel
    reader_train = DataSetReader(X_train, y_train)
    reader_val = DataSetReader(X_val, y_val)
    reader_test = DataSetReader(X_test, y_test)

    return reader_train, reader_val, reader_test
 