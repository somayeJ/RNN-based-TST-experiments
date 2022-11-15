from __future__ import unicode_literals
import os
import sys
import argparse
import pprint
import time
import random
import numpy as np
import tensorflow as tf
from vocab import Vocabulary, build_vocab
from sklearn.metrics import confusion_matrix
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--model_type',
            type=str,
            default='') # 'transformer' or 'seq2seq'
    argparser.add_argument('--mode',
            type=str,
            default='') # 'rec', 'tsf' or ''
    argparser.add_argument('--train',
            type=str,
            default='')
    argparser.add_argument('--dev',
            type=str,
            default='')
    argparser.add_argument('--test',
            type=str,
            default='')
    argparser.add_argument('--vocab',
            type=str,
            default='')
    argparser.add_argument('--embedding',
            type=str,
            default='')
    argparser.add_argument('--model',
            type=str,
            default='')
    argparser.add_argument('--load_model',
            type=bool,
            default=False)
    argparser.add_argument('--batch_size',
            type=int,
            default=64)
    argparser.add_argument('--max_epochs',
            type=int,
            default=20)
    argparser.add_argument('--steps_per_checkpoint',
            type=int,
            default=1000)
    argparser.add_argument('--max_seq_length',
            type=int,
            default=20)
    argparser.add_argument('--max_train_size',
            type=int,
            default=-1)
    argparser.add_argument('--dropout_keep_prob',
            type=float,
            default=0.5)
    argparser.add_argument('--n_layers',
            type=int,
            default=1)
    argparser.add_argument('--dim_emb',
            type=int,
            default=100)
    argparser.add_argument('--learning_rate',
            type=float,
            default=0.0005)
    argparser.add_argument('--rho',                 # loss_rec + rho * loss_adv
            type=float,
            default=1)
    argparser.add_argument('--gamma_init',          # softmax(logit / gamma)
            type=float,
            default=0.1)
    argparser.add_argument('--gamma_decay',
            type=float,
            default=1)
    argparser.add_argument('--gamma_min',
            type=float,
            default=0.1)
    argparser.add_argument('--filter_sizes',
            type=str,
            default='1,2,3,4,5')
    argparser.add_argument('--n_filters',
            type=int,
            default=128)
    args = argparser.parse_args()
    print ('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print ('------------------------------------------------')
    return args

def load_sent(path, max_size=-1):
    '''
    :param path:
    :param max_size:
    :return: # a list of lists of sentence-tokens for each input sequence
    '''
    data = []
    with open(path) as f:
        for line in f:
            if len(data) == max_size:
                break
            data.append(line.split())
    return data

def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)

def cnn(inp, filter_sizes, n_filters, dropout, scope, reuse=False):
    '''
    :param inp:
    :param filter_sizes: default 1,2,3,4,5
    :param n_filters: default 128
    :param dropout:
    :param scope:
    :param reuse:
    :return:
    '''
    dim = inp.get_shape().as_list()[-1]
    inp = tf.expand_dims(inp, -1)

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()
        outputs = []
        for size in filter_sizes:
            with tf.variable_scope('conv-maxpool-%s' % size):
                W = tf.get_variable('W', [size, dim, 1, n_filters])
                b = tf.get_variable('b', [n_filters])
                conv = tf.nn.conv2d(inp, W,
                    strides=[1, 1, 1, 1], padding='VALID')

                #conv = tf.nn.conv2d(inp, W,
                    #strides=[1, 1, 1, 1], padding='SAME')
                h = leaky_relu(conv + b)
                # max pooling over time
                pooled = tf.reduce_max(h, reduction_indices=1)
                pooled = tf.reshape(pooled, [-1, n_filters])
                outputs.append(pooled)
        outputs = tf.concat(outputs, 1)
        outputs = tf.nn.dropout(outputs, dropout)

        with tf.variable_scope('output'):
            W = tf.get_variable('W', [n_filters*len(filter_sizes), 1])
            b = tf.get_variable('b', [1])
            logits = tf.reshape(tf.matmul(outputs, W) + b, [-1])
    return logits


class Model(object):
    def __init__(self, args, vocab):
        dim_emb = args.dim_emb
        filter_sizes = [int(x) for x in args.filter_sizes.split(',')] # default 1,2,3,4,5
        n_filters = args.n_filters # default 128

        self.dropout = tf.placeholder(tf.float32,
                                      name='dropout')
        self.learning_rate = tf.placeholder(tf.float32,
                                            name='learning_rate')
        self.x = tf.placeholder(tf.int32, [None, None],  # batch_size * max_len
                                name='x')
        self.y = tf.placeholder(tf.float32, [None],
                                name='y')
        embedding = tf.get_variable('embedding', [vocab.size, dim_emb])
        x = tf.nn.embedding_lookup(embedding, self.x)
        self.logits = cnn(x, filter_sizes, n_filters, self.dropout, 'cnn')
        self.probs = tf.sigmoid(self.logits)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.y, logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.loss)
        self.saver = tf.train.Saver()


def create_model(sess, args, vocab):
    model = Model(args, vocab)
    if args.load_model:
        print 'Loading model from', args.model
        model.saver.restore(sess, args.model)
    else:
        print 'Creating model with fresh parameters.'
        sess.run(tf.global_variables_initializer())
    return model


def evaluate(sess, args, vocab, model, x, y):
    probs = []
    batches = get_batches(x, y, vocab.word2id, args.batch_size)
    for batch in batches:
        p = sess.run(model.probs,
                     feed_dict={model.x: batch['x'],
                                model.dropout: 1})
        probs += p.tolist()
    y_hat = [p > 0.5 for p in probs]
    same = [p == q for p, q in zip(y, y_hat)]
    if args.test:
        cf= confusion_matrix(y, y_hat)
        tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
        print(cf)
        print('0:negative, 1:positive''tn, fp, fn, tp',tn, fp, fn, tp)
    return 100.0 * sum(same) / len(y), probs


def get_batches(x, y, word2id, batch_size, min_len=5):
    pad = word2id['<pad>']
    unk = word2id['<unk>']

    batches = []
    s = 0
    while s < len(x):
        t = min(s + batch_size, len(x))
        _x = []
        max_len = max([len(sent) for sent in x[s:t]])
        max_len = max(max_len, min_len)
        for sent in x[s:t]:
            sent_id = [word2id[w] if w in word2id else unk for w in sent]
            padding = [pad] * (max_len - len(sent))
            _x.append(padding + sent_id)
        batches.append({'x': _x,
                        'y': y[s:t]})
        s = t
    return batches


def makeup(_x, n):
    x = []
    for i in range(n):
        # % : baghimande taghsim
        x.append(_x[i % len(_x)])
    return x


def prepare(path, args, suffix=''):
    '''
    :param path:
    :param suffix:
    :return: train_x (consisting of all sentences of the 2 corpora), train_y: all the labels, sorted based on their len
    '''
    #data0_amazon = load_sent('../data/amazon/binary/sentiment.train.0')
    #data1_amazon = load_sent('../data/amazon/binary/sentiment.train.1')
    if args.model_type == 'transformer':
        data0 = load_sent(path + 'test.' + suffix  + '.0' + '.txt') # data0 aval - boode , tu style transfer files yani .tsf files + hast
        data1 = load_sent(path + 'test.' + suffix  + '.1' + '.txt')
    elif args.model_type == 'seq2seq':
        data0 = load_sent(path + 'sentiment.test.0.' + suffix) # data0 aval - boode , tu style transfer files yani .tsf files + hast
        data1 = load_sent(path + 'sentiment.test.1.' + suffix)
    # data balancing part(me):
    if path[-3:]=='ain':
        print('train', len(data0), len(data1))
        if len(data0) < len(data1):
            data0 = makeup(data0, len(data1))
        elif len(data1) < len(data0):
            data1 = makeup(data1, len(data0))
        print('train_balanced',len(data0), len(data1))
    elif path[-3:] == 'dev':
        print('dev', len(data0), len(data1))

    elif path[-3:] == 'est':
        print(' test', len(data0), len(data1))
    if suffix == 'tsf':
        x = data0 + data1
        y = [1] * len(data0) + [0] * len(data1)
    else :
        x = data0 + data1
        y = [0] * len(data0) + [1] * len(data1)

    '''
    x = data0 + data0_amazon  + data1 + data1_amazon
    y = [0] * (len(data0) + len(data0_amazon)) + [1] * (len(data1) + len(data1_amazon))
    '''
    z = sorted(zip(x, y), key=lambda i: len(i[0]))
    return zip(*z)

if __name__ == '__main__':
    args = load_arguments()
    if args.train:
        train_x, train_y = prepare(args.train, args, args.mode)

        if not os.path.isfile(args.vocab):
            build_vocab(train_x, args.vocab)

    vocab = Vocabulary(args.vocab)
    print 'vocabulary size', vocab.size

    if args.dev:
        dev_x, dev_y = prepare(args.dev, args, args.mode)

    if args.test:
        test_x, test_y = prepare(args.test, args, args.mode)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, args, vocab)
        if args.train:
            batches = get_batches(train_x, train_y,
                                  vocab.word2id, args.batch_size)
            random.shuffle(batches)

            start_time = time.time()
            step = 0
            loss = 0.0
            best_dev = float('-inf')
            learning_rate = args.learning_rate

            for epoch in range(1, 1 + args.max_epochs):
                print '--------------------epoch %d--------------------' % epoch

                for batch in batches:
                    step_loss, _ = sess.run([model.loss, model.optimizer],
                                            feed_dict={model.x: batch['x'],
                                                       model.y: batch['y'],
                                                       model.dropout: args.dropout_keep_prob,
                                                       model.learning_rate: learning_rate})

                    step += 1
                    loss += step_loss / args.steps_per_checkpoint

                    if step % args.steps_per_checkpoint == 0:
                        print 'step %d, time %.0fs, loss %.2f' \
                              % (step, time.time() - start_time, loss)
                        loss = 0.0

                if args.dev:
                    acc, _ = evaluate(sess, args, vocab, model, dev_x, dev_y)
                    print 'dev accuracy %.2f' % acc
                    if acc > best_dev:
                        best_dev = acc
                        print 'Saving model...'
                        model.saver.save(sess, args.model)

        if args.test:
            acc, _ = evaluate(sess, args, vocab, model, test_x, test_y)
            print 'test accuracy %.2f' % acc
