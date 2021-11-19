from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense, Flatten, Reshape, Multiply, Permute
from tensorflow.keras.layers import Conv1D, Convolution2D
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import ThresholdedReLU, LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import tensorflow as tf

# tf.enable_eager_execution()

# tf_config = tf.ConfigProto() 
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4

alpha_var = K.variable(0.5)

def adaf_3(x, alpha = 0.5): 
    return K.maximum(x, 0.0) * K.exp(-x * alpha)

def adaf_3_opt(x, alpha = alpha_var):
    return K.maximum(x, 0.0) * K.exp(-x * alpha + 1.0)

def leaky_adaf_3_opt(x, alpha = alpha_var, leak = 0.01):
    return leak * K.minimum(x, 0.0) + K.maximum(x, 0.0) * K.exp(-x * alpha + 1.0)

def leaky_adaf_3(x, alpha=1.0, leak = 0.01):
    return leak * K.minimum(x, 0.0) + K.maximum(x, 0.0) * K.exp(-x * alpha)

import pdb

class LrnLeakyAdaf(Layer):
    def __init__(self, alpha, name, **kwargs):
        super(LrnLeakyAdaf, self).__init__(**kwargs)
        # pdb.set_trace()
        print('calling init')
        self.alpha = alpha
        self.__name__ = name

    def build_(self, input_shape):
        print('calling build')
        super(LrnLeakyAdaf, self).build(input_shape)

    def call(self, x):
        leak = 0.01
        return leak * K.minimum(x, 0.0) + K.maximum(x, 0.0) * K.exp(-x * self.alpha)

    def get_config(self):
        # config = {'alpha': K.cast(self.alpha, dtype='float32')}
        config = {'alpha': self.alpha.value}
        print('alpha', self.alpha, self.alpha.value)
        base_config = super(LrnLeakyAdaf, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

get_custom_objects().update({'lrn_leaky_adaf_3': LrnLeakyAdaf(K.variable(0.5), 'lr_leaky')})
get_custom_objects().update({'adaf_3': Activation(adaf_3)})
get_custom_objects().update({'leaky_adaf_3': Activation(leaky_adaf_3)})
get_custom_objects().update({'leaky_adaf_3_opt': Activation(leaky_adaf_3_opt)})
get_custom_objects().update({'adaf_3_opt': Activation(adaf_3_opt)})

class Metrics(Callback):
  def on_train_begin(self, logs={}):
    self.val_f1s_weighted = []
    self.val_recalls_weighted = []
    self.val_precisions_weighted = []

    self.val_f1s_macro = []
    self.val_recalls_macro = []
    self.val_precisions_macro = []

  def evaluate_f1s(self, val_predict, val_targ):
    _val_f1_weighted = f1_score(val_targ, val_predict, average='weighted')
    _val_recall_weighted = recall_score(val_targ, val_predict, average='weighted')
    _val_precision_weighted = precision_score(val_targ, val_predict, average='weighted')
    self.val_f1s_weighted.append(_val_f1_weighted)
    self.val_recalls_weighted.append(_val_recall_weighted)
    self.val_precisions_weighted.append(_val_precision_weighted)

    _val_f1_macro = f1_score(val_targ, val_predict, average='macro')
    _val_recall_macro = recall_score(val_targ, val_predict, average='macro')
    _val_precision_macro = precision_score(val_targ, val_predict, average='macro')
    self.val_f1s_macro.append(_val_f1_macro)
    self.val_recalls_macro.append(_val_recall_macro)
    self.val_precisions_macro.append(_val_precision_macro)

    print(" — [weighted] val_f1: %f — val_precision: %f — val_recall %f" % (_val_f1_weighted, _val_precision_weighted, _val_recall_weighted))
    print(" — [MACRO] val_f1: %f — val_precision: %f — val_recall %f" % (_val_f1_macro, _val_precision_macro, _val_recall_macro))
    return

  def on_epoch_end(self, epoch, logs={}): 
    print('--> ALPHA = ', K.get_value(alpha_var))
    val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
    val_targ = self.validation_data[1]
    self.evaluate_f1s(val_predict, val_targ)
    return

class CharCNNZhang(object):
  """
  Class to implement the Character Level Convolutional Neural Network for Text Classification,
  as described in Zhang et al., 2015 (http://arxiv.org/abs/1509.01626)
  """
  def __init__(self, input_size, alphabet_size, embedding_size,
                 conv_layers, fully_connected_layers, num_of_classes,
                 threshold, dropout_p,
                 optimizer='adam', loss='categorical_crossentropy'):
    """
    Initialization for the Character Level CNN model.
    Args:
        input_size (int): Size of input features
        alphabet_size (int): Size of alphabets to create embeddings for
        embedding_size (int): Size of embeddings
        conv_layers (list[list[int]]): List of Convolution layers for model
        fully_connected_layers (list[list[int]]): List of Fully Connected layers for model
        num_of_classes (int): Number of classes in data
        threshold (float): Threshold for Thresholded ReLU activation function
        dropout_p (float): Dropout Probability
        optimizer (str): Training optimizer
        loss (str): Loss function
    """
    self.input_size = input_size
    self.alphabet_size = alphabet_size
    self.embedding_size = embedding_size
    self.conv_layers = conv_layers
    self.fully_connected_layers = fully_connected_layers
    self.num_of_classes = num_of_classes
    self.threshold = threshold
    self.dropout_p = dropout_p
    self.optimizer = optimizer
    self.loss = loss
    self._build_model(type = 1)  # 1 = standard neurons, 2 = pynada

  def _squeeze_and_excitation_layer(self, input_data, ratio):
    out_dim = int(input_data.shape[-1])
    squeeze = GlobalAveragePooling1D()(input_data)
    squeeze = Reshape((-1,out_dim))(squeeze)

    excitation = Dense(int(out_dim / ratio), activation="relu")(squeeze)
    excitation = Dense(out_dim, activation="sigmoid")(excitation)

    scale = Multiply()([input_data, excitation])

    return scale

  def _build_model(self, type = 1):
    """
    Build and compile the Character Level CNN model

    Returns: None
    """
    # Input layer
    inputs = Input(shape=(self.input_size,), name='sent_input', dtype='int64')

    # Embedding layers
    x = Embedding(self.alphabet_size + 1, self.embedding_size, input_length=self.input_size)(inputs)
    x = Reshape((5000, 256))(x)
   
    idx = 0
    # Convolution layers
    for cl in self.conv_layers:
      idx += 1
      if type == 1:
        x = Conv1D(cl[0], cl[1])(x)

        if cl[2] == 0:
          x = ThresholdedReLU(self.threshold)(x)
        elif cl[2] == 1:
          x = Activation(adaf_3_opt)(x)
        elif cl[2] == 2:
          x = Activation(leaky_adaf_3_opt)(x)

        if cl[3] != -1:
          x = MaxPooling1D(cl[3], cl[3])(x)

        if cl[4] != -1:
          x = self._squeeze_and_excitation_layer(input_data = x, ratio = cl[4])

      elif type == 2:
        x1 = Conv1D(cl[0], cl[1])(x)
        x1 = ThresholdedReLU(self.threshold)(x1)

        x2 = Conv1D(cl[0], cl[1])(x)
        if cl[2] == 1:
          x2 = Activation(adaf_3_opt)(x2)
        elif cl[2] == 2:
          x2 = Activation(leaky_adaf_3_opt)(x2)

        x = tf.keras.layers.Add()([x1, x2])

        if cl[3] != -1:
          x = MaxPooling1D(cl[3], cl[3])(x)

        if cl[4] != -1:
          x = self._squeeze_and_excitation_layer(input_data = x, ratio = cl[4])

    # Flatten the features
    x = Flatten()(x)

    # Fully connected layers
    for fl in self.fully_connected_layers:
      if type == 1:
        x = Dense(fl)(x)
        x = ThresholdedReLU(self.threshold)(x)
      elif type == 2:
        x1 = Dense(fl)(x)
        x1 = ThresholdedReLU(self.threshold)(x1)
        x2 = Dense(fl)(x)
        x2 = Activation(leaky_adaf_3_opt)(x2)

        x = tf.keras.layers.Add()([x1, x2])
      
    x = Dropout(self.dropout_p)(x)

    # Output layer
    predictions = Dense(self.num_of_classes, activation="softmax")(x)

    # Build and compile the model
    model = Model(inputs=inputs, outputs=predictions)
    
    optimizer = tf.keras.optimizers.Adam(lr=0.0005)
    
    model.layers[-1].trainable_weights.extend([alpha_var])

    # load weights
    # model.load_weights("checkpoints/weights-improvement-02-acc-0.69-loss-0.58.hdf5")
    # print('Alpha is', K.get_value(alpha_var))

    model.compile(optimizer=optimizer, loss=self.loss, metrics=['accuracy'])
    self.model = model

    print("CharCNNZhang model build: ")
    self.model.summary()

  def train(self, training_inputs, training_labels,
              validation_inputs, validation_labels,
              epochs, batch_size, checkpoint_every=100):
    """
    Training function
    Args:
        training_inputs (numpy.ndarray): Training set inputs
        training_labels (numpy.ndarray): Training set labels
        validation_inputs (numpy.ndarray): Validation set inputs
        validation_labels (numpy.ndarray): Validation set labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        checkpoint_every (int): Interval for logging to Tensorboard
    Returns: None
    """
    # Create callbacks
    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=checkpoint_every, batch_size=batch_size,
    #                           write_graph=False, write_grads=True, write_images=False,
    #                           embeddings_freq=checkpoint_every,
    #                           embeddings_layer_names=None)
    # Start training

    filepath="checkpoints/weights-improvement-{epoch:02d}-acc-{val_acc:.2f}-loss-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    metrics = Metrics()
    metrics.validation_data = (validation_inputs, validation_labels)
    callbacks_list = [metrics, checkpoint]

    print("Training CharCNNZhang model: ")
    self.model.fit(training_inputs, training_labels,
                    validation_data=(validation_inputs, validation_labels),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=callbacks_list)

  def test(self, testing_inputs, testing_labels, batch_size, output_txt):
    """
    Testing function
    Args:
        testing_inputs (numpy.ndarray): Testing set inputs
        testing_labels (numpy.ndarray): Testing set labels
        batch_size (int): Batch size
    Returns: None
    """
    # Evaluate inputs
    results = self.model.evaluate(testing_inputs, testing_labels, batch_size=batch_size, verbose=1)
    predicts = self.model.predict(testing_inputs, batch_size=batch_size, verbose=1)
    
    np.savetxt(output_txt, np.argmax(predicts, axis=1))

    return results

  def save(self, file_path):
    self.model.save(file_path)
