import tensorflow as tf
from train import *


def adaf_v3(x, alpha=0.3):
   y = tf.math.maximum(x, 0) * tf.exp(-x * alpha)
   return y


def leaky_adaf_3(x, alpha=0.9, leak=0.01):
    y = leak * tf.math.minimum(x, 0) + tf.math.maximum(x, 0) * tf.exp(-x * alpha)
    return y

def swish(x, alpha):
    y = x * tf.nn.sigmoid(alpha * x)
    return y
    
def rbf_bump(x, alpha):
   return tf.exp(-(x ** 2) / alpha)


exp = Experiment(leaky_adaf_3, 'leaky_0.9')
exp.run()
