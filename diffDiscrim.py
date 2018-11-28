import math
import numpy as np
import tensorflow as tf
import glob
import sys
from sys import stdout
import os
import time

from tensorflow.python.framework import ops

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image/255 * 2 - 1
def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return ((image + 1) / 2)*255


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def conv2d(input_, output_dim,
           k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02,
           name="conv2d",pad="SAME"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        if pad=="VALID":
            conv = tf.pad(input_, [[0,0],[1,1],[1,1],[0,0]], mode="CONSTANT")
            conv = tf.nn.conv2d(conv, w, strides=[1, d_h, d_w, 1], padding=pad)
        else:
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=pad)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        return conv

class DiffDiscrim(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, df_dim=16,
                 input_c_dim=3,
                 checkpoint_dir=None, data=None, momentum=0.9):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size

        self.input_c_dim = input_c_dim
        self.df_dim = df_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1', momentum=momentum)
        # self.d_bn2 = batch_norm(name='d_bn2', momentum=momentum)
        # self.d_bn3 = batch_norm(name='d_bn3', momentum=momentum)


        self.data = data
        self.checkpoint_dir = checkpoint_dir

        data_description = data.get_data_description()
        # Create an iterator for the data

        data_description = data.get_data_description()
        data_description = [data_description[0], {
            key: [None, *description]
            for key, description in data_description[1].items()}]

        self.iter_handle = tf.placeholder(tf.string, shape=[],
                                              name='training_placeholder')
        iterator = tf.data.Iterator.from_string_handle(
            self.iter_handle, *data_description)
        training_batch = iterator.get_next()

        # self.build_model(training_batch['labels'], training_batch['rgb'])
        self.build_model(training_batch['labels'],training_batch['pos'],training_batch['neg'])

    def build_model(self, target, pos, neg):
        # TODO check if preprocess to -1 1 is necessary
        self.target_placeholder = preprocess(target)
        self.pos_placeholder = preprocess(pos)
        self.neg_placeholder = preprocess(neg)

        PosExample = tf.concat([self.target_placeholder, self.pos_placeholder], 3)
        NegExample = tf.concat([self.target_placeholder, self.neg_placeholder], 3)

        self.D, self.D_logits = self.discriminator(PosExample)
        self.D_, self.D_logits_ = self.discriminator(NegExample,reuse=True)

        self.d_loss_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))

        self.d_loss = self.d_loss_pos + self.d_loss_neg

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.target_sum = tf.summary.image("Target", target[...,::-1])
        self.positiv_sum = tf.summary.image("Positiv", pos[...,::-1])
        self.negativ_sum = tf.summary.image("Negativ", neg[...,::-1])

        self.d_loss_pos_sum = tf.summary.scalar("d_loss_pos", self.d_loss_pos)
        self.d_loss_neg_sum = tf.summary.scalar("d_loss_neg", self.d_loss_neg)

        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]

        self.saver = tf.train.Saver()


    def train(self, args):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # TODO check Momentum vs Adam
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.sum = tf.summary.merge([self.d__sum,self.target_sum,self.positiv_sum,
            self.negativ_sum, self.d_loss_pos_sum, self.d_sum, self.d_loss_neg_sum, self.d_loss_sum])

        self.writer = tf.summary.FileWriter(self.checkpoint_dir, self.sess.graph)

        if args.checkpoint is not None and self.load(os.path.join(args.EXP_OUT,str(args.checkpoint))):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        input_data = self.data.get_trainset()
        data_iterator = input_data.repeat(args.max_epochs).batch(args.batch_size).make_one_shot_iterator()
        data_handle = self.sess.run(data_iterator.string_handle())
        globalCounter = 1
        localCounter = 1
        start_time = time.time()

        while True:
            if np.mod(globalCounter, args.num_print) == 1:
                try:
                    _, summary_str, d_l = self.sess.run([d_optim, self.sum, self.d_loss],
                                                   feed_dict={ self.iter_handle: data_handle })
                except tf.errors.OutOfRangeError:
                    print("INFO: Done with all steps")
                    self.save(self.checkpoint_dir, globalCounter)
                    break

                self.writer.add_summary(summary_str, globalCounter-1)
                print("Step: [%2d] rate: %4.4f steps/sec, d_loss: %.8f" \
                    % (globalCounter*args.batch_size,args.batch_size*localCounter/(time.time() - start_time), d_l))
                stdout.flush()

                mean_val_D, mean_val_D_ = self.validation(args, out=True, loaded=True)
                print("Mean Validation: Same: %f \t Diff: %f \t Abs: %f" % (mean_val_D,mean_val_D_,1-mean_val_D+mean_val_D_))
                stdout.flush()
                start_time = time.time()
                localCounter = 1
            else:
                try:
                    self.sess.run(d_optim,feed_dict={ self.iter_handle: data_handle })
                except tf.errors.OutOfRangeError:
                    print("INFO: Done with all training steps")
                    self.save(self.checkpoint_dir, globalCounter)
                    break
            globalCounter += 1
            localCounter += 1

        self.validation(args, loaded=True)

    def validation(self, args, out=False, loaded=False):
        if not loaded:
            if args.checkpoint is not None and self.load(os.path.join(args.EXP_OUT,str(args.checkpoint))):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                raise ValueError('Could not load checkpoint and that is needed for validation')
        args.batch_size=1
        input_data = self.data.get_validation_set()
        data_iterator = input_data.repeat(1).batch(1).make_one_shot_iterator()
        data_handle = self.sess.run(data_iterator.string_handle())
        counter = 1
        start_time = time.time()
        pred_array = np.zeros((15,2))
        while True:
            try:
                D, D_ = self.sess.run([self.D, self.D_],
                                               feed_dict={ self.iter_handle: data_handle })
                pred_array[counter-1,:] = [np.mean(D),np.mean(D_)]
                if not out:
                    print("Validation image %d: Same: %f \t Diff: %f" % (counter, np.mean(D),np.mean(D_)))
            except tf.errors.OutOfRangeError:
                break

            counter += 1
        return np.mean(pred_array, axis=0)

    def discriminator(self, image, y=None, reuse=False):
        # image is 256 x 256 x (input_c_dim + input_c_dim)
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, k_h=4, k_w=4, d_h=4, d_w=4, name='d_h0_conv'))
            # h0 is (64 x 64 x self.df_dim)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, k_h=4, k_w=4, d_h=4, d_w=4, name='d_h1_conv')))
            # h1 is (16 x 16 x self.df_dim*2)
            h2 = conv2d(h1, 1, k_h=2, k_w=2, d_h=2, d_w=2, name='d_h2_conv')
            # h2 is (30 x 30 x 1)
            # print(h2.shape)

            return tf.nn.sigmoid(h2), h2

    def save(self, checkpoint_dir, step):
        model_name = "diffDiscrim.model"
        # model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        # checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        # if not os.path.exists(checkpoint_dir):
        #     os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        self.saver.restore(self.sess, checkpoint)
        self.graph = tf.get_default_graph()
        return True