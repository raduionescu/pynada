import tensorflow as tf
import numpy as np
import datetime
import os
import sys
import re
import pdb
from sklearn.metrics import confusion_matrix

from data_set_reader import create_readers
import utils
import model

operating_system = sys.platform
print(operating_system, operating_system.find("win"))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if operating_system.find("win") == -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

# do not delete this
logs_folder = 'logs'
RUNNING_ID = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
utils.set_vars(logs_folder, RUNNING_ID)
utils.create_dir(logs_folder)


reader_train, reader_val, reader_test = create_readers()


class Experiment:

    def __init__(self, activation_fn, name, learning_rate_init=10 ** -3, step_decay=15, num_epochs=30, batch_size=64):
        self.name = name + "mlp_2_lr_%.5f_step_%d_b_%d" % (learning_rate_init, step_decay, batch_size)
        self.learning_rate_init = learning_rate_init
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_folder = "checkpoints_" + self.name
        self.IS_RESTORE = tf.train.latest_checkpoint(self.checkpoint_folder) is not None
        self.inputs_ = tf.placeholder(np.float32, [None, 784])
        self.targets_ = tf.placeholder(np.float32, [None, 10])

        # build neural network
        # TODO: here you can change the architecture
        self.logits = model.mlp_2(self.inputs_, activation_fn)
        self.cost = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets_)
        self.avg_cost = tf.reduce_mean(self.cost)
        self.global_step = tf.Variable(0, trainable=False)  # 782
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_init, self.global_step,
                                                        step_decay * (50000 // batch_size), 0.1, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.avg_cost,
                                                                                           global_step=self.global_step)
        self.sess = tf.Session(config=config)

        self.train_loss_placeholder = tf.placeholder(tf.float32, shape=[], name="train_loss")
        self.val_loss_placeholder = tf.placeholder(tf.float32, shape=[], name="val_loss")
        self.test_loss_placeholder = tf.placeholder(tf.float32, shape=[], name="test_loss")

        self.train_acc_placeholder = tf.placeholder(tf.float32, shape=[], name="train_acc")
        self.val_acc_placeholder = tf.placeholder(tf.float32, shape=[], name="val_acc")
        self.test_acc_placeholder = tf.placeholder(tf.float32, shape=[], name="test_acc")

        tf.summary.scalar('train_loss', self.train_loss_placeholder)
        tf.summary.scalar('val_loss', self.val_loss_placeholder)
        tf.summary.scalar('test_loss', self.test_loss_placeholder)

        tf.summary.scalar('train_acc', self.train_acc_placeholder)
        tf.summary.scalar('val_acc', self.val_acc_placeholder)
        tf.summary.scalar('test_acc', self.test_acc_placeholder)
        tf.summary.scalar('learning_rate', self.learning_rate)
        self.merged = tf.summary.merge_all()

    def fit(self, epoch):
        iters = int(np.ceil(reader_train.num_samples / self.batch_size))
        total_loss = 0
        un_norm_acc = 0
        for iter in range(iters):
            batch_x, batch_y = reader_train.next_batch(self.batch_size)
            _, c, predictions, _, lr = self.sess.run([self.optimizer, self.cost, self.logits, self.global_step,
                                                      self.learning_rate],
                                                     feed_dict={self.inputs_: batch_x,  self.targets_: batch_y})

            total_loss += np.sum(c)
            un_norm_acc += np.sum(np.argmax(predictions, axis=1) == np.argmax(batch_y, axis=1))
        
        return total_loss / reader_train.num_samples, un_norm_acc / reader_train.num_samples

    def eval(self, reader, return_predicted_labels=False):
        iters = int(np.ceil(reader.num_samples / self.batch_size))
        total_loss = 0
        un_norm_acc = 0
        if return_predicted_labels:
            pred_labels = None

        for iter in range(iters):
            batch_x, batch_y = reader.next_batch(self.batch_size)
            c, predictions = self.sess.run([self.cost, self.logits],
                                            feed_dict={self.inputs_: batch_x,  self.targets_: batch_y})
            total_loss += np.sum(c)
            un_norm_acc += np.sum(np.argmax(predictions, axis=1) == np.argmax(batch_y, axis=1))
            if return_predicted_labels:
                if pred_labels is None:
                    pred_labels = np.argmax(predictions, axis=1)
                else:
                    pred_labels = np.concatenate((pred_labels, np.argmax(predictions, axis=1)))

        if return_predicted_labels:
            return total_loss / reader.num_samples, un_norm_acc / reader.num_samples, pred_labels
        else:
            return total_loss / reader.num_samples, un_norm_acc / reader.num_samples

    def restore_model(self, epoch=None):
        if epoch is None:
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_folder)
        else:
            checkpoint_path = os.path.join(self.checkpoint_folder, "model_%d" % epoch)

        if checkpoint_path is None:
            raise Exception("Checkpoint file is missing!")

        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=0)
        saver.restore(self.sess, checkpoint_path)

    def get_statistics_set(self, reader, epoch=None):
        self.restore_model(epoch=epoch)
        loss, acc, predicted_labels = self.eval(reader, return_predicted_labels=True)
        utils.log_message('loss = {}, acc = {} \nconf mat = \n{}'.format(loss, acc, confusion_matrix(np.argmax(reader.labels, axis=1), predicted_labels)))

    def get_statistics(self, epoch=None):
        utils.log_message("Statistics for epoch: {}".format(epoch))
        utils.log_message("TRAINING")
        self.get_statistics_set(reader_train, epoch)
        utils.log_message("VAL")
        self.get_statistics_set(reader_val, epoch)
        utils.log_message("TEST")
        self.get_statistics_set(reader_test, epoch)

    def run(self):
        start_epoch = 0
        saver = tf.train.Saver(max_to_keep=0)
        self.sess.run(tf.global_variables_initializer())
        if self.IS_RESTORE:
            print('=' * 30 + '\nRestoring from ' + tf.train.latest_checkpoint(self.checkpoint_folder))
            saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_folder))
            start_epoch = re.findall(r'\d+', tf.train.latest_checkpoint(self.checkpoint_folder))
            start_epoch = int(start_epoch[-1]) + 1

        writer = tf.summary.FileWriter('train_%s.log' % self.name, self.sess.graph)

        for epoch in range(start_epoch, self.num_epochs):
            utils.log_message("Epoch: %d/%d" % (epoch, self.num_epochs))
            train_loss, train_acc = self.fit(epoch)
            val_loss, val_acc = self.eval(reader_val)
            test_loss, test_acc = self.eval(reader_test)
            utils.log_message("acc train = {}, val = {}, test = {}.".format(train_acc, val_acc, test_acc))
            utils.log_message("loss train = %.4f, val = %.4f, test = %.4f" % (train_loss, val_loss, test_loss))
            merged_ = self.sess.run(self.merged, feed_dict={
                                                            self.train_loss_placeholder: train_loss,
                                                            self.val_loss_placeholder: val_loss,
                                                            self.test_loss_placeholder: test_loss,
                                                            self.train_acc_placeholder: train_acc,
                                                            self.val_acc_placeholder: val_acc,
                                                            self.test_acc_placeholder: test_acc})
            writer.add_summary(merged_, epoch)

            saver.save(self.sess, os.path.join(self.checkpoint_folder, "model_%d" % epoch))


