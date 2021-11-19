
import tensorflow as tf
from train import *


exp = Experiment(tf.nn.leaky_relu, 'relu_leaky')
exp.run()

