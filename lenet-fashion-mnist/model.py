import tensorflow as tf
import pdb


def neuron_lenet_alpha_opt(input_, activation_fn):
    print('Using PyNADA with learnable alpha')
    # 32 x 32 x 1
    alpha_init = 0.7
    alpha = tf.Variable(alpha_init, name="alpha")
    conv_1 = tf.layers.conv2d(inputs=input_, filters=6, kernel_size=(5, 5), padding='valid', activation=None)
    conv_1 = activation_fn(conv_1, alpha)

    conv_1_relu = tf.layers.conv2d(inputs=input_, filters=6, kernel_size=(5, 5), padding='valid', activation=None)
    conv_1_relu = tf.nn.relu(conv_1_relu)

    # 28 x 28 x 6
    max_pool_1 = tf.layers.max_pooling2d(conv_1 + conv_1_relu, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 14 x 14 x 6
    conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=16, kernel_size=(5, 5), padding='valid', activation=None)
    conv_2 = activation_fn(conv_2, alpha)

    conv_2_relu = tf.layers.conv2d(inputs=max_pool_1, filters=16, kernel_size=(5, 5), padding='valid', activation=None)
    conv_2_relu = tf.nn.relu(conv_2_relu)

    # 10 x 10 x 16
    max_pool_2 = tf.layers.max_pooling2d(conv_2 + conv_2_relu, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 5 x 5 x 16
    flat = tf.layers.flatten(max_pool_2)
    fc_1 = tf.layers.dense(flat, units=120, activation=None)
    fc_1 = activation_fn(fc_1, alpha)
    fc_1_relu = tf.layers.dense(flat, units=120, activation=tf.nn.relu)

    fc_2 = tf.layers.dense(fc_1 + fc_1_relu, units=84, activation=None)
    fc_2 = activation_fn(fc_2, alpha)
    fc_2_relu = tf.layers.dense(fc_1 + fc_1_relu, units=84, activation=tf.nn.relu)

    logits = tf.layers.dense(fc_2 + fc_2_relu, units=10, activation=None)
    return logits


def lenet_alpha(input_, activation_fn):
    print('Using ADA with learnable alpha')
    # 32 x 32 x 1
    alpha_init = 0.1
    alpha_1 = tf.Variable(alpha_init, name="alpha_1")
    conv_1 = tf.layers.conv2d(inputs=input_, filters=6, kernel_size=(5, 5), padding='valid', activation=None)
    conv_1 = activation_fn(conv_1, alpha_1)
    # 28 x 28 x 6
    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 14 x 14 x 6
    conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=16, kernel_size=(5, 5), padding='valid', activation=None)
    conv_2 = activation_fn(conv_2, alpha_1)
    # 10 x 10 x 16
    max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 5 x 5 x 16
    flat = tf.layers.flatten(max_pool_2)
    fc_1 = tf.layers.dense(flat, units=120, activation=tf.nn.relu)
    fc_2 = tf.layers.dense(fc_1, units=84, activation=tf.nn.relu)
    logits = tf.layers.dense(fc_2, units=10, activation=None)
    return logits


def lenet(input_, activation_fn):
    print('Using ADA or RELU.')
    # 32 x 32 x 1
    conv_1 = tf.layers.conv2d(inputs=input_, filters=6, kernel_size=(5, 5), padding='valid', activation=None)
    conv_1 = activation_fn(conv_1)
    # 28 x 28 x 6
    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 14 x 14 x 6
    conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=16, kernel_size=(5, 5), padding='valid', activation=None)
    conv_2 = activation_fn(conv_2)
    # 10 x 10 x 16
    max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 5 x 5 x 16
    flat = tf.layers.flatten(max_pool_2)
    fc_1 = tf.layers.dense(flat, units=120, activation=tf.nn.relu)
    fc_2 = tf.layers.dense(fc_1, units=84, activation=tf.nn.relu)
    logits = tf.layers.dense(fc_2, units=10, activation=None)
    return logits


def neuron_lenet(input_, activation_fn):
    print('Using PyNADA.')
    # 32 x 32 x 1
    conv_1 = tf.layers.conv2d(inputs=input_, filters=6, kernel_size=(5, 5), padding='valid', activation=None)
    conv_1 = activation_fn(conv_1)

    conv_1_relu = tf.layers.conv2d(inputs=input_, filters=6, kernel_size=(5, 5), padding='valid', activation=None)
    conv_1_relu = tf.nn.relu(conv_1_relu)

    # 28 x 28 x 6
    max_pool_1 = tf.layers.max_pooling2d(conv_1 + conv_1_relu, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 14 x 14 x 6
    conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=16, kernel_size=(5, 5), padding='valid', activation=None)
    conv_2 = activation_fn(conv_2)

    conv_2_relu = tf.layers.conv2d(inputs=max_pool_1, filters=16, kernel_size=(5, 5), padding='valid', activation=None)
    conv_2_relu = tf.nn.relu(conv_2_relu)

    # 10 x 10 x 16
    max_pool_2 = tf.layers.max_pooling2d(conv_2 + conv_2_relu, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 5 x 5 x 16
    flat = tf.layers.flatten(max_pool_2)
    fc_1 = tf.layers.dense(flat, units=120, activation=activation_fn)
    fc_1_relu = tf.layers.dense(flat, units=120, activation=tf.nn.relu)

    fc_2 = tf.layers.dense(fc_1 + fc_1_relu, units=84, activation=activation_fn)
    fc_2_relu = tf.layers.dense(fc_1 + fc_1_relu, units=84, activation=tf.nn.relu)

    logits = tf.layers.dense(fc_2 + fc_2_relu, units=10, activation=None)
    return logits


def lenet_for_relu(input_, activation_fn):
    print('Using RELU.')
    # 32 x 32 x 1
    conv_1 = tf.layers.conv2d(inputs=input_, filters=6, kernel_size=(5, 5), padding='valid', activation=activation_fn)
    # 28 x 28 x 6
    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 14 x 14 x 6
    conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=16, kernel_size=(5, 5), padding='valid', activation=activation_fn)
    # 10 x 10 x 16
    max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 5 x 5 x 16
    flat = tf.layers.flatten(max_pool_2)
    fc_1 = tf.layers.dense(flat, units=120, activation=activation_fn)
    fc_2 = tf.layers.dense(fc_1, units=84, activation=activation_fn)
    logits = tf.layers.dense(fc_2, units=10, activation=None)
    return logits

