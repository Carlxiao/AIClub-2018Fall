#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os


from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.dataflow import dataset

import tensorflow as tf

"""
Train a CNN on MNIST.
Get to 99.33% test accuracy.
"""

BATCH_SIZE = 128
NUM_UNITS = None


class Model(ModelDesc):
    """
    A simple convnet for MNIST
    The main structure is drew from: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    """
    def __init__(self):
        super(Model, self).__init__()

    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 28, 28], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        assert tf.test.is_gpu_available()
        image = tf.expand_dims(image, 3)
        l = image * 2 - 1   # center the pixels values at zero

        l = Conv2D('conv1', l, 32, 3, activation=tf.nn.relu)
        l = Conv2D('conv2', l, 64, 3, activation=tf.nn.relu)
        l = MaxPooling('pool2', l, 2)
        l = Dropout(l, keep_prob=0.75)
        l = FullyConnected('fc1', l, 128)
        l = tf.nn.relu(l)
        l = Dropout(l, keep_prob=0.5)
        logits = FullyConnected('fc2', l, 10)

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        add_moving_summary(cost)

        wrong = tf.to_float(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), name='wrong_vector')
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        return tf.identity(cost, name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        return opt


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Mnist(train_or_test)
    # ds = dataset.Mnist(train_or_test, dir='data')
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default=0)
    # parser.add_argument('--load', help='load model for training')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir()

    dataset_train = get_data('train')
    dataset_test = get_data('test')

    config = AutoResumeTrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test, [ScalarStats('cost'), ClassificationError('wrong_vector')]),
            ScheduledHyperParamSetter('learning_rate', [(1, 1e-3), (30, 1e-4), (40, 1e-5)])
        ],
        max_epoch=60
        # , session_init=SaverRestore(args.load) if args.load else None
    )
    launch_train_with_config(config, SimpleTrainer())
