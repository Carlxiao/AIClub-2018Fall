#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os


from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary, add_tensor_summary
from tensorpack.dataflow import dataset
from tensorpack.tfutils import optimizer, gradproc
from utils import TensorPrinterEpoch

import tensorflow as tf

"""
Generate adversarial samples for a CNN on MNIST.
"""

N = 4
BATCH_SIZE = N * N
NUM_UNITS = None


class Model(ModelDesc):
    """
    A simple convnet for MNIST
    The main structure is drew from: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    """
    def __init__(self, batch):
        super(Model, self).__init__()
        self.image = batch[0]
        self.label = tf.constant(9, tf.int32) - batch[1]    # the wrong labels

    def inputs(self):
        # return [tf.placeholder(tf.float32, [None, 28, 28], 'input'), tf.placeholder(tf.int32, [None], 'label')]
        return []

    def build_graph(self):
        assert tf.test.is_gpu_available()

        image_raw = tf.constant(self.image, tf.float32)
        noise = tf.get_variable('noise', shape=[BATCH_SIZE, 28, 28], initializer=tf.random_normal_initializer(stddev=0.01))
        image = tf.add(image_raw, noise, name='image')
        image = tf.clip_by_value(image, -1e-5, 1 + 1e-5)
        noise_show = tf.concat([tf.concat([noise[i*N+j] for j in range(N)], axis=1) for i in range(N)], axis=0, name='noise_show')
        image_show = tf.concat([tf.concat([image[i*N+j] for j in range(N)], axis=1) for i in range(N)], axis=0, name='image_show')
        l = tf.expand_dims(image, 3)
        l = l * 2 - 1   # center the pixels values at zero

        l = Conv2D('conv1', l, 32, 3, activation=tf.nn.relu)
        l = Conv2D('conv2', l, 64, 3, activation=tf.nn.relu)
        l = MaxPooling('pool2', l, 2)
        # l = Dropout(l, keep_prob=0.75)
        l = FullyConnected('fc1', l, 128)
        l = tf.nn.relu(l)
        # l = Dropout(l, keep_prob=0.5)
        logits = FullyConnected('fc2', l, 10)
        tf.argmax(logits, 1, name='pred_label')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        add_tensor_summary(cost, ['scalar'])
        # l1_cost = regularize_cost('noise', tf.contrib.layers.l1_regularizer(1e-2))
        l2_cost = regularize_cost('noise', tf.contrib.layers.l2_regularizer(1e-2))
        add_tensor_summary(l2_cost, ['scalar'])
        add_tensor_summary(tf.div(tf.reduce_sum(tf.abs(noise)), BATCH_SIZE, name='L1Norm'), ['scalar'])
        add_tensor_summary(tf.div(tf.sqrt(tf.reduce_sum(tf.square(noise))), BATCH_SIZE, name='L2Norm'), ['scalar'])

        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, self.label, 1)), name='wrong_vector')
        add_tensor_summary(tf.reduce_mean(wrong, name='train_error'), ['scalar'])

        return tf.add_n([cost, l2_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('lr', initializer=3e-3, trainable=False)
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.AdamOptimizer(lr)
        # freeze all variables in network
        opt = optimizer.apply_grad_processors(opt, [gradproc.ScaleGradient([('.*/b', 0.), ('.*/W', 0.)])])
        return opt


def get_data(train_or_test='train'):
    isTrain = train_or_test == 'train'
    # ds = dataset.Mnist(train_or_test)
    ds = dataset.Mnist(train_or_test, shuffle=False)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default=0)
    parser.add_argument('--load', help='load model for training', default='train_log/train/checkpoint')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir()

    dataset = get_data('train')
    dataset.reset_state()
    batches = [batch for batch in dataset]

    config = TrainConfig(
        model=Model(batches[1]),
        dataflow=DataFromList([[]]),
        callbacks=[
            # ModelSaver(),
            TensorPrinterEpoch(['pred_label']),
            DumpTensorAsImage('image_show', 'img'),
            DumpTensorAsImage('noise_show', 'noise'),
            ScheduledHyperParamSetter('lr', [(20, 1e-3)])
        ],
        max_epoch=30,
        steps_per_epoch=200,
        session_init=SaverRestore(args.load)
    )
    launch_train_with_config(config, SimpleTrainer())
