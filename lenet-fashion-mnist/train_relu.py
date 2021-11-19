
import tensorflow as tf
from train import *


exp = Experiment(tf.nn.relu, 'relu_local')
exp.run()

