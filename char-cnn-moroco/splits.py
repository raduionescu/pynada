import os
import numpy as np

TRAIN      = '1'
VALIDATION = '2'
TEST       = '3'
PRIVATE    = '4'

class Data(object):
  def __init__(self, dataset, alphabet, input_size, num_of_classes):
    """
    Initialization of a Data object.
    Args:
        dataset (str): Raw dataset
        alphabet (str): Alphabet of characters to index
        input_size (int): Size of input features
        num_of_classes (int): Number of classes in data
    """
    self.alphabet = alphabet
    self.alphabet_size = len(self.alphabet)
    self.dict = {}  # Maps each character to an integer
    self.no_of_classes = num_of_classes

    for idx, char in enumerate(alphabet):
        self.dict[char] = idx + 1

    self.length = input_size
    self.dataset = dataset

  def convert_data(self):
    """
    Return all loaded data from data variable.
    Returns:
        (np.ndarray) Data transformed from raw to indexed form with associated one-hot label.
    """

    one_hot = np.eye(self.no_of_classes, dtype='int64')
    classes = []
    batch_indices = []

    for text, label in self.dataset:
      batch_indices.append(self.str_to_indexes(text))
      classes.append(one_hot[int(label) - 1])

    return np.asarray(batch_indices, dtype="int64"), np.asarray(classes)

  def str_to_indexes(self, text):
    """
    Convert a string to character indexes based on character dictionary.

    Args:
        s (str): String to be converted to indexes
    Returns:
        str2idx (np.ndarray): Indexes of characters in s
    """
    max_length = min(len(text), self.length)
    str2idx = np.zeros(self.length, dtype="int64")
    for i in range(1, max_length + 1):
      char = text[-i]
      if char in self.dict:
        str2idx[i - 1] = self.dict[char]

    return str2idx


def read_data(path):
  return open(path, 'r', encoding="utf8").read().split("\n")

def generate_splits(dataset, indices):
  data = list(zip(dataset, indices))

  train_data       = list(filter(lambda record: record[1] == TRAIN     , data))
  validation_data  = list(filter(lambda record: record[1] == VALIDATION, data))
  test_data        = list(filter(lambda record: record[1] == TEST      , data))
  private_data     = list(filter(lambda record: record[1] == PRIVATE   , data))

  train_data       = list(map(lambda d: d[0], train_data))
  validation_data  = list(map(lambda d: d[0], validation_data))
  test_data        = list(map(lambda d: d[0], test_data))
  private_data     = list(map(lambda d: d[0], private_data))

  return (train_data, validation_data, test_data, private_data)

def get_category(klass, category, get_originals = True):
  sufix = ".original" if get_originals else ""

  originals = read_data("../dataset/{}/{}{}.txt".format(klass, category, sufix))
  indices   = read_data("../dataset/{}/{}_split.txt".format(klass, category))

  return generate_splits(originals, indices)

def get_dataset(klass, get_originals = True, except_category = None):
  categories = [
    "culture",
    "finance",
    "politics",
    "science",
    "sport",
    "tech"
  ]

  train      = []
  validation = []
  test       = []
  private    = []

  for category in categories:
    if(except_category != None and except_category == category):
      print("Skipping category {}".format(category))
      continue

    data = get_category(klass, category, get_originals)

    train      = train      + data[0]
    validation = validation + data[1]
    test       = test       + data[2]
    private    = private    + data[3]

  return (train, validation, test, private)

def append_category_to_dataset(all_data, category):
  return list(map(lambda data: list(map(lambda d: (d, category), data)), all_data))
