import tensorflow as tf
import pdb


def mlp_2_opt(input_, activation_fn):
    alpha = tf.Variable(1.0, name="alpha")
    fc_1 = tf.layers.dense(input_, units=100, activation=None)
    fc_1 = activation_fn(fc_1, alpha)
    fc_2 = tf.layers.dense(fc_1, units=10, activation=None)
    fc_2 = activation_fn(fc_2, alpha)
    logits = tf.layers.dense(fc_2, units=10, activation=None)
    return logits


def mlp_2_neuron_opt(input_, activation_fn):
    alpha = tf.Variable(1.0, name="alpha")
    fc_1 = tf.layers.dense(input_, units=100, activation=None)
    fc_1 = activation_fn(fc_1, alpha)
    fc_1_relu = tf.layers.dense(input_, units=100, activation=tf.nn.relu)

    fc_2 = tf.layers.dense(fc_1 + fc_1_relu, units=10, activation=None)
    fc_2 = activation_fn(fc_2, alpha)
    fc_2_relu = tf.layers.dense(fc_1 + fc_1_relu, units=10, activation=tf.nn.relu)

    logits = tf.layers.dense(fc_2 + fc_2_relu, units=10, activation=None)
    return logits


def mlp_2(input_, activation_fn):
    fc_1 = tf.layers.dense(input_, units=100, activation=activation_fn)
    fc_2 = tf.layers.dense(fc_1, units=10, activation=activation_fn)
    logits = tf.layers.dense(fc_2, units=10, activation=None)
    return logits


def mlp_2_neuron(input_, activation_fn):
    fc_1 = tf.layers.dense(input_, units=100, activation=activation_fn)
    fc_1_relu = tf.layers.dense(input_, units=100, activation=tf.nn.relu)

    fc_2 = tf.layers.dense(fc_1 + fc_1_relu, units=10, activation=activation_fn)
    fc_2_relu = tf.layers.dense(fc_1 + fc_1_relu, units=10, activation=tf.nn.relu)

    logits = tf.layers.dense(fc_2 + fc_2_relu, units=10, activation=None)
    return logits


def mlp_1_neuron_opt(input_, activation_fn):
    alpha = tf.Variable(0.1, name="alpha")
    fc_1 = tf.layers.dense(input_, units=100, activation=None)
    z_1 = activation_fn(fc_1, alpha)
    z_2 = tf.layers.dense(input_, units=100, activation=tf.nn.relu)
    logits = tf.layers.dense(z_1 + z_2, units=10, activation=None)
    return logits


def mlp_1_neuron(input_, activation_fn):
    fc_1 = tf.layers.dense(input_, units=100, activation=None)
    z_1 = activation_fn(fc_1)
    z_2 = tf.layers.dense(input_, units=100, activation=tf.nn.relu)

    logits = tf.layers.dense(z_1 + z_2, units=10, activation=None)
    return logits


def mlp_1_opt(input_, activation_fn):
    alpha = tf.Variable(1.0, name="alpha")
    fc_1 = tf.layers.dense(input_, units=100, activation=None)
    fc_1 = activation_fn(fc_1, alpha)
    logits = tf.layers.dense(fc_1, units=10, activation=None)
    return logits


def mlp_1(input_, activation_fn):
    fc_1 = tf.layers.dense(input_, units=100, activation=activation_fn)
    logits = tf.layers.dense(fc_1, units=10, activation=None)
    return logits


