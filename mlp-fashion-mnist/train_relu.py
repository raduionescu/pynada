
import tensorflow as tf
from train import *


exp = Experiment(tf.nn.leaky_relu, 'leaky_relu_', num_epochs=30)
exp.run()
