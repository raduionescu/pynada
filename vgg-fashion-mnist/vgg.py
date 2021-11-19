import tensorflow as tf
import pdb


def __add_conv(input_, num_filters, activation_fn, alpha, kernel_size=(3, 3)):

    layer = tf.layers.conv2d(inputs=input_, filters=num_filters, kernel_size=kernel_size, padding='same',
                             activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    # layer = activation_fn(layer, alpha) # use this in order to have a learnable alpha
    layer = activation_fn(layer)
    return layer


def __add_conv_neuron(input_, num_filters, activation_fn, alpha, kernel_size=(3, 3)):

    adaf = tf.layers.conv2d(inputs=input_, filters=num_filters, kernel_size=kernel_size, padding='same',
                            activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    # adaf = activation_fn(adaf, alpha) # use this in order to have a learnable alpha
    adaf = activation_fn(adaf)
    relu = tf.layers.conv2d(inputs=input_, filters=num_filters, kernel_size=kernel_size, padding='same',
                            activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    return adaf + relu


def __add_fc(input_, num_units, activation_fn, alpha):
    layer = tf.layers.dense(input_, units=num_units, activation=None,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    if activation_fn is None:
        return layer
    # layer = activation_fn(layer, alpha) # use this in order to have a learnable alpha
    layer = activation_fn(layer)
    return layer


def __add_fc_neuron(input_, num_units, activation_fn, alpha):
    adaf = tf.layers.dense(input_, units=num_units, activation=None,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    # adaf = activation_fn(adaf, alpha) # use this in order to have a learnable alpha
    adaf = activation_fn(adaf)
    relu = tf.layers.dense(input_, units=num_units, activation=tf.nn.relu,
                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
    return adaf + relu


def vgg9(inputs_, activation_fn, is_training):
    alpha = tf.Variable(0.5)
    # block 1
    conv_1 = __add_conv(inputs_, 64, activation_fn, alpha)
    pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=[2, 2], strides=2, padding="same")
    drop_1 = tf.layers.dropout(pool_1, 0.1, training=is_training)

    # block 2
    conv_2 = __add_conv(drop_1, 128, activation_fn, alpha)
    pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=[2, 2], strides=2, padding="same")
    drop_2 = tf.layers.dropout(pool_2, 0.1, training=is_training)

    # block 3
    conv_3 = __add_conv(drop_2, 256, activation_fn, alpha)
    conv_4 = __add_conv(conv_3, 256, activation_fn, alpha)
    pool_3 = tf.layers.max_pooling2d(inputs=conv_4, pool_size=[2, 2], strides=2, padding="same")
    drop_3 = tf.layers.dropout(pool_3, 0.2, training=is_training)

    # block 4
    conv_5 = __add_conv(drop_3, 512, activation_fn, alpha)
    conv_6 = __add_conv(conv_5, 512, activation_fn, alpha)
    pool_4 = tf.layers.max_pooling2d(inputs=conv_6, pool_size=[2, 2], strides=2, padding="same")
    drop_4 = tf.layers.dropout(pool_4, 0.3, training=is_training)
    
    # block 5
    conv_7 = __add_conv(drop_4, 512, activation_fn, alpha)
    conv_8 = __add_conv(conv_7, 512, activation_fn, alpha)
    pool_5 = tf.layers.max_pooling2d(inputs=conv_8, pool_size=[2, 2], strides=2, padding="same")
    drop_5 = tf.layers.dropout(pool_5, 0.3, training=is_training)    

    flat = tf.layers.flatten(drop_5)  
    logits = __add_fc(flat, 10, None, None)

    return logits, alpha


def vgg9_neuron(inputs_, activation_fn, is_training):
    alpha = tf.Variable(1.0)
    # block 1
    conv_1 = __add_conv_neuron(inputs_, 64, activation_fn, alpha)
    pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=[2, 2], strides=2, padding="same")
    drop_1 = tf.layers.dropout(pool_1, 0.1, training=is_training)

    # block 2
    conv_2 = __add_conv_neuron(drop_1, 128, activation_fn, alpha)
    pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=[2, 2], strides=2, padding="same")
    drop_2 = tf.layers.dropout(pool_2, 0.1, training=is_training)

    # block 3
    conv_3 = __add_conv_neuron(drop_2, 256, activation_fn, alpha)
    conv_4 = __add_conv_neuron(conv_3, 256, activation_fn, alpha)
    pool_3 = tf.layers.max_pooling2d(inputs=conv_4, pool_size=[2, 2], strides=2, padding="same")
    drop_3 = tf.layers.dropout(pool_3, 0.2, training=is_training)

    # block 4
    conv_5 = __add_conv_neuron(drop_3, 512, activation_fn, alpha)
    conv_6 = __add_conv_neuron(conv_5, 512, activation_fn, alpha)
    pool_4 = tf.layers.max_pooling2d(inputs=conv_6, pool_size=[2, 2], strides=2, padding="same")
    drop_4 = tf.layers.dropout(pool_4, 0.3, training=is_training)

    # block 5
    conv_7 = __add_conv_neuron(drop_4, 512, activation_fn, alpha)
    conv_8 = __add_conv_neuron(conv_7, 512, activation_fn, alpha)
    pool_5 = tf.layers.max_pooling2d(inputs=conv_8, pool_size=[2, 2], strides=2, padding="same")
    drop_5 = tf.layers.dropout(pool_5, 0.3, training=is_training)

    flat = tf.layers.flatten(drop_5)  
    logits = __add_fc(flat, 10, None, None)

    return logits, alpha
    

