# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np 
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorpack import *
from tensorpack.tfutils import summary, optimizer
from tensorpack.tfutils.gradproc import GlobalNormClip

args = None
BATCH_SIZE = 128
GRADIENT_CLIP = 5

class CharRNNData(RNGDataFlow):
    def __init__(self, data_file, seq_length, size=100000):
        self.size = size
        self.seq_length = seq_length
        print("Loading data...")
        with open(data_file, 'r', encoding='utf-8') as f:
            data = f.readlines()
            self.text = ''.join(data)
            del data
        print("Building vocabulary...")
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=5000, filters='', char_level=True)
        self.tokenizer.fit_on_texts(self.text)
        self.whole_seq = np.array(
            self.tokenizer.texts_to_sequences([self.text])[0], np.int32)
        self.vocab_size = len(self.tokenizer.word_index)
        print("Vocabulary size: {}".format(self.vocab_size))
    
    def __len__(self):
        return self.size 
    
    def __iter__(self):
        random_starts = self.rng.randint(
            0, self.whole_seq.shape[0] - self.seq_length - 1, (self.size,))
        for st in random_starts:
            seq = self.whole_seq[st:st + self.seq_length + 1]
            yield [seq[:-1], seq[1:]]
        
class Model(ModelDesc):
    def __init__(self, num_rnn_layer, num_rnn_unit, seq_len, vocab_size):
        self.num_rnn_layer = num_rnn_layer
        self.num_rnn_unit = num_rnn_unit
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def inputs(self):
        return [tf.placeholder(tf.int32, (None, self.seq_len), 'input'),
                tf.placeholder(tf.int32, (None, self.seq_len), 'nextinput')]

    def build_graph(self, input, nextinput):
        # embedding
        embeddingW = tf.get_variable('embedding', [self.vocab_size, self.num_rnn_unit])
        input_feature = tf.nn.embedding_lookup(embeddingW, input)
        input_list = tf.unstack(input_feature, axis=1)

        # rnn
        cell = rnn.MultiRNNCell([rnn.LSTMBlockCell(num_units=self.num_rnn_unit)
                                for _ in range(self.num_rnn_layer)])

        def get_v(n):
            ret = tf.get_variable(n + '_unused', [BATCH_SIZE, self.num_rnn_unit],
                                  trainable=False,
                                  initializer=tf.constant_initializer())
            ret = tf.placeholder_with_default(ret, shape=[None, self.num_rnn_unit], name=n)
            return ret
        initial_state = [rnn.LSTMStateTuple(get_v('c{}'.format(i)), get_v('h{}'.format(i))) 
                            for i in range(self.num_rnn_layer)]

        outputs, last_state = rnn.static_rnn(cell, input_list, initial_state)
        last_state = tf.identity(last_state, 'last_state')

        # FC
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.num_rnn_unit])
        logits = FullyConnected('fc', output, self.vocab_size)
        tf.nn.softmax(logits, name='prob')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.reshape(nextinput, [-1]))
        cost = tf.reduce_mean(loss, name='cost')
        summary.add_moving_summary(cost)
        return cost

    def optimizer(self):
        lr = tf.get_variable(
            'learning_rate', initializer=2e-2, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        return optimizer.apply_grad_processors(opt, [GlobalNormClip(GRADIENT_CLIP)])

def generate(data_file, model_path, start, length):
    ds = CharRNNData(data_file, args.seq_len)
    input_names = ['input'] + ['c{}'.format(i) for i in range(args.num_rnn_layer)] + \
                  ['h{}'.format(i) for i in range(args.num_rnn_layer)]
    pred = OfflinePredictor(PredictConfig(
        model=Model(args.num_rnn_layer, args.num_rnn_unit, args.seq_len, ds.vocab_size),
        session_init=SaverRestore(model_path),
        input_names=input_names,
        output_names=['prob', 'last_state']))

    # feed the starting chars
    initial = np.zeros([1, args.num_rnn_unit])
    for char in start[-1]:
        x = np.array([[ds.tokenizer.word_index[char]]], dtype='int32')
        _, state = pred(*([x] + [initial] * args.num_rnn_layer * 2))
    
    # generate
    ret = start 
    char = start[-1]
    for _ in range(length):
        x = np.array([[ds.tokenizer.word_index[char]]], dtype='int32')
        prob, state = pred(*([x] + [state[i, 0] for i in range(args.num_rnn_layer)] +
                             [state[i, 1] for i in range(args.num_rnn_layer)]))
        # randomly sample a char according to the distribution of prob
        idx = int(np.searchsorted(np.cumsum(prob),
                                  np.random.rand(1) * (np.sum(prob) - 0.2) + 0.2))
        char = ds.tokenizer.index_word[idx]
        ret += char
    print(ret)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='model to load')
    parser.add_argument('--text', help='training text data (in UTF-8)')
    parser.add_argument('--num_rnn_layer', default=2, type=int, help='number of layers in RNN')
    parser.add_argument('--num_rnn_unit', default=64, type=int, help='number of units in RNN')
    parser.add_argument('--seq_len', default=11, type=int, help='sequence length')
    parser.add_argument('--generate', action='store_true', help='to generate text')
    parser.add_argument('--gen_len', default=100, type=int, help='length of text to generate')
    parser.add_argument('--start', default='一', help='length of text to generate')
    parser.add_argument('--logdir', help='log directory')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.generate:
        assert args.load is not None
        args.seq_len = 1
        generate(args.text, args.load, args.start, args.gen_len)
    else:
        ds = CharRNNData(args.text, args.seq_len)
        # logger.auto_set_dir(name=args.logdir)
        logger.set_logger_dir('train_log/' + args.logdir)
        config = AutoResumeTrainConfig(
            data=QueueInput(BatchData(ds, BATCH_SIZE)),
            callbacks=[
                # ScheduledHyperParamSetter('learning_rate', [(5, 2e-4)]),
                ModelSaver()
            ],
            model=Model(args.num_rnn_layer, args.num_rnn_unit, args.seq_len, ds.vocab_size),
            max_epoch=100,
            # steps_per_epoch=100,
            session_init=SaverRestore(args.load) if args.load else None
        )
        launch_train_with_config(config, SimpleTrainer())

