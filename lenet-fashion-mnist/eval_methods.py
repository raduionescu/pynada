import tensorflow as tf
import sys

from train import *
import utils


def adaf(x, alpha=1.0):
    y = tf.math.maximum(x, 0) * tf.exp(-x * alpha)
    return y


epoch = None
if len(sys.argv) > 1:
    epoch = int(sys.argv[1])

exp_1 = Experiment(adaf, 'adaf')
utils.log_message("ADAF")
exp_1.get_statistics(epoch)

tf.reset_default_graph()
exp_2 = Experiment(tf.nn.relu, 'relu')
utils.log_message("RELU")
exp_2.get_statistics(epoch)