import argparse
import pprint
import os
import sys
import time
import random
import cPickle as pickle
import numpy as np
import tensorflow as tf
from vocab import Vocabulary, build_vocab
from utils import *
from nn import *
import generation_method
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    # indicating the path to where to save and load the data and models
    argparser.add_argument('--train',
            type=str,
            default='../../data/yelp/sentiment.train')
    argparser.add_argument('--dev',
            type=str,
            default='../../data/yelp/sentiment.dev')
    argparser.add_argument('--test',
            type=str,
            default='../../data/yelp/sentiment.test')
    argparser.add_argument('--output',
            type=str,
            default='./outputs/sentiment.dev') # while training and testing filenames are sentiment.dev & sentiment.test 
    argparser.add_argument('--vocab',
            type=str,
            default='./outputs/yelp.vocab')
    argparser.add_argument('--embedding',
            type=str,
            default='') # path to the pretrained embedding to initialize the tokens with
    argparser.add_argument('--online_testing',
            type=bool,
            default=False)
    argparser.add_argument('--load_model',
            type=bool,
            default=False) # --load_model False if you want to train a model from scratch
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
    argparser.add_argument('--beam',
            type=int,
            default=1) # if --beam 1 greedy search technique & if --beam >1, beam technique is used to generate the text
    argparser.add_argument('--dropout_keep_prob',
            type=float,
            default=0.5)
    argparser.add_argument('--n_layers',
            type=int,
            default=1)
    argparser.add_argument('--dim_y',
            type=int,
            default=200)
    argparser.add_argument('--dim_z',
            type=int,
            default=500)
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

class Model(object):
    def __init__(self, args, vocab):
        dim_y = args.dim_y # 200
        dim_z = args.dim_z
        dim_h = dim_y + dim_z
        dim_emb = args.dim_emb
        n_layers = args.n_layers
        max_len = args.max_seq_length
        filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
        n_filters = args.n_filters
        beta1, beta2 = 0.5, 0.999
        grad_clip = 30.0
        self.dropout = tf.placeholder(tf.float32,
            name='dropout')
        self.learning_rate = tf.placeholder(tf.float32,
            name='learning_rate')
        self.rho = tf.placeholder(tf.float32,
            name='rho')
        self.gamma = tf.placeholder(tf.float32,
            name='gamma')
        self.batch_len = tf.placeholder(tf.int32,
            name='batch_len')
        self.batch_size = tf.placeholder(tf.int32,
            name='batch_size')
        self.enc_inputs = tf.placeholder(tf.int32, [None, None],    #size * len
            name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, None],
            name='dec_inputs')
        self.targets = tf.placeholder(tf.int32, [None, None],
            name='targets')
        self.weights = tf.placeholder(tf.float32, [None, None],
            name='weights')
        self.labels = tf.placeholder(tf.float32, [None],
            name='labels')

        labels = tf.reshape(self.labels, [-1, 1])
        embedding = tf.get_variable('embedding',
            initializer=vocab.embedding.astype(np.float32))

        with tf.variable_scope('projection0'):
            proj_W0 = tf.get_variable('W', [dim_h, vocab.size])
            proj_b0 = tf.get_variable('b', [vocab.size])

        with tf.variable_scope('projection1'):
            proj_W1 = tf.get_variable('W', [dim_h, vocab.size])
            proj_b1 = tf.get_variable('b', [vocab.size])

        enc_inputs = tf.nn.embedding_lookup(embedding, self.enc_inputs)
        dec_inputs = tf.nn.embedding_lookup(embedding, self.dec_inputs)

        #####   auto-encoder   #####
        init_state = tf.concat([linear(labels, dim_y, scope='encoder'),
            tf.zeros([self.batch_size, dim_z])], 1)
        print(init_state)
        # create GRU cell
        cell_e = create_cell(dim_h, n_layers, self.dropout) # dim_h = dim_y +dim_z =700
        _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
            initial_state=init_state, scope='encoder') #
        z = z[:, dim_y:]

        half = self.batch_size / 2

        self.h_ori0 = tf.concat([linear(labels[:half], dim_y,
            scope='generator0'), z[:half]], 1) #rec0
        self.h_ori1 = tf.concat([linear(labels[half:], dim_y,
            scope='generator1'), z[half:]], 1) #rec1

        self.h_tsf0 = tf.concat([linear(1-labels[:half], dim_y,
            scope='generator1', reuse=True), z[:half]], 1) #tsf0
        self.h_tsf1 = tf.concat([linear(1-labels[half:], dim_y,
            scope='generator0', reuse=True), z[half:]], 1) #tsf1

        cell_g0 = create_cell(dim_h, n_layers, self.dropout)
        g_outputs0, _ = tf.nn.dynamic_rnn(cell_g0, dec_inputs[:half],
            initial_state=self.h_ori0, scope='generator0') #rx0

        cell_g1 = create_cell(dim_h, n_layers, self.dropout)
        g_outputs1, _ = tf.nn.dynamic_rnn(cell_g1, dec_inputs[half:],
            initial_state=self.h_ori1, scope='generator1') #rx1

        teach_h0 = tf.concat([tf.expand_dims(self.h_ori0, 1), g_outputs0], 1)
        teach_h1 = tf.concat([tf.expand_dims(self.h_ori1, 1), g_outputs1], 1)

        g_outputs0 = tf.nn.dropout(g_outputs0, self.dropout)
        g_outputs0 = tf.reshape(g_outputs0, [-1, dim_h])
        g_logits0 = tf.matmul(g_outputs0, proj_W0) + proj_b0

        g_outputs1 = tf.nn.dropout(g_outputs1, self.dropout)
        g_outputs1 = tf.reshape(g_outputs1, [-1, dim_h])
        g_logits1 = tf.matmul(g_outputs1, proj_W1) + proj_b1

        # Computes sparse softmax cross entropy 
        loss_rec0 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets[:half], [-1]), logits=g_logits0)
        loss_rec0 *= tf.reshape(self.weights[:half], [-1])
        self.loss_rec0 = tf.reduce_sum(loss_rec0) / tf.to_float(self.batch_size)

        loss_rec1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets[half:], [-1]), logits=g_logits1)
        loss_rec1 *= tf.reshape(self.weights[half:], [-1])
        self.loss_rec1 = tf.reduce_sum(loss_rec1) / tf.to_float(self.batch_size)
        self.loss_rec = self.loss_rec0 + self.loss_rec1
        #####   feed-previous decoding   #####
        go = dec_inputs[:,0,:]

        soft_func0 = softsample_word(self.dropout, proj_W0, proj_b0, embedding,
            self.gamma)
        soft_func1 = softsample_word(self.dropout, proj_W1, proj_b1, embedding,
            self.gamma)

        hard_func0 = argmax_word(self.dropout, proj_W0, proj_b0, embedding)
        hard_func1 = argmax_word(self.dropout, proj_W1, proj_b1, embedding)

        # rnn_decode returns sequence of outputs
        soft_h_ori0, soft_logits_ori0 = rnn_decode(self.h_ori0, go[:half], max_len,
            cell_g0, soft_func0, scope='generator0') # soft_rec0
        # generating tsf(transferred) and rec(reconstructed) forms of x
        soft_h_tsf1, soft_logits_tsf1 = rnn_decode(self.h_tsf1, go[half:], max_len,
            cell_g0, soft_func0, scope='generator0') # soft_tsf1
        hard_h_ori0, self.hard_logits_ori0 = rnn_decode(self.h_ori0, go[:half], max_len,
            cell_g0, hard_func0, scope='generator0') # hard_rec0
        hard_h_tsf1, self.hard_logits_tsf1 = rnn_decode(self.h_tsf1, go[half:], max_len,
            cell_g0, hard_func0, scope='generator0') # hard_tsf1

        soft_h_ori1, soft_logits_ori1 = rnn_decode(self.h_ori1, go[half:], max_len,
            cell_g1, soft_func1, scope='generator1') # soft_rec1
        soft_h_tsf0, soft_logits_tsf0 = rnn_decode(self.h_tsf0, go[:half], max_len,
            cell_g1, soft_func1, scope='generator1') # soft_tsf0
        hard_h_ori1, self.hard_logits_ori1 = rnn_decode(self.h_ori1, go[half:], max_len,
            cell_g1, hard_func1, scope='generator1') # hard_rec1
        hard_h_tsf0, self.hard_logits_tsf0 = rnn_decode(self.h_tsf0, go[:half], max_len,
            cell_g1, hard_func1, scope='generator1') # hard_tsf0

        #####   discriminator   #####
        # a batch's first half consists of sentences of one style and second half of the other
        zeros, ones = self.labels[:half], self.labels[half:]
        soft_h_tsf0 = soft_h_tsf0[:, :1+self.batch_len, :]
        soft_h_tsf1 = soft_h_tsf1[:, :1+self.batch_len, :]

        self.loss_d0, self.loss_g0 = discriminator(teach_h0, soft_h_tsf1,
            ones, zeros, filter_sizes, n_filters, self.dropout,
            scope='discriminator0')
        self.loss_d1, self.loss_g1 = discriminator(teach_h1, soft_h_tsf0,
            ones, zeros, filter_sizes, n_filters, self.dropout,
            scope='discriminator1')
        #####   optimizer   #####
        self.loss_adv = self.loss_g0 + self.loss_g1
        self.loss0 = self.loss_rec0 + self.rho * self.loss_g0
        self.loss1 = self.loss_rec1 + self.rho * self.loss_g1
        self.loss = self.loss0 + self.loss1

        theta_eg0 = retrive_var(['encoder', 'generator0',
            'embedding', 'projection0'])
        theta_eg1 = retrive_var(['encoder', 'generator1',
            'embedding', 'projection1'])

        theta_d0 = retrive_var(['discriminator0'])
        theta_d1 = retrive_var(['discriminator1'])

        opt = tf.train.AdamOptimizer(self.learning_rate, beta1, beta2)

        grad_rec0, _ = zip(*opt.compute_gradients(self.loss_rec0, theta_eg0))
        grad_rec1, _ = zip(*opt.compute_gradients(self.loss_rec1, theta_eg1))
        grad_adv0, _ = zip(*opt.compute_gradients(self.loss_g0, theta_eg0))
        grad_adv1, _ = zip(*opt.compute_gradients(self.loss_g1, theta_eg1))

        grad0, _ = zip(*opt.compute_gradients(self.loss0, theta_eg0))
        grad0, _ = tf.clip_by_global_norm(grad0, grad_clip)
        grad1, _ = zip(*opt.compute_gradients(self.loss1, theta_eg1))
        grad1, _ = tf.clip_by_global_norm(grad1, grad_clip)

        self.grad_rec_norm0 = tf.global_norm(grad_rec0)
        self.grad_rec_norm1 = tf.global_norm(grad_rec1)
        self.grad_adv_norm0 = tf.global_norm(grad_adv0)
        self.grad_adv_norm1 = tf.global_norm(grad_adv1)
        self.grad_norm0 = tf.global_norm(grad0)
        self.grad_norm1 = tf.global_norm(grad1)

        self.optimize_rec0 = opt.minimize(self.loss_rec0, var_list=theta_eg0)
        self.optimize_rec1 = opt.minimize(self.loss_rec1, var_list=theta_eg1)

        self.optimize_tot0 = opt.apply_gradients(zip(grad0, theta_eg0))
        self.optimize_tot1 = opt.apply_gradients(zip(grad1, theta_eg1))

        self.optimize_d0 = opt.minimize(self.loss_d0, var_list=theta_d0)
        self.optimize_d1 = opt.minimize(self.loss_d1, var_list=theta_d1)

        self.saver = tf.train.Saver()
        

class Accumulator(object):
    def __init__(self, div, names):
        self.div = div
        self.names = names
        self.n = len(self.names)
        self.values = [0] * self.n

    def clear(self):
        self.values = [0] * self.n

    def add(self, values):
        for i in range(self.n):
            self.values[i] += values[i] / self.div

    def output(self, s=''):
        if s:
            s += ' '
        for i in range(self.n):
            s += '%s %.2f' % (self.names[i], self.values[i])
            if i < self.n-1:
                s += ', '
        print (s)
    def output_list(self):
        return self.values, self.names

def transfer(model, decoder, sess, args, vocab, data0, data1, out_path):
    batches, order0, order1 = get_batches(data0, data1,
        vocab.word2id, args.batch_size,args)

    data0_rec, data1_rec = [], []
    data0_tsf, data1_tsf = [], []
    losses = Accumulator(len(batches), ['loss', 'rec', 'adv', 'd0', 'd1'])
    for batch in batches:
        rec0, tsf0, rec1, tsf1  = decoder.rewrite(batch)

        data0_rec += rec0
        data1_rec += rec1
        data0_tsf += tsf0
        data1_tsf += tsf1

        loss, loss_rec, loss_adv, loss_d0, loss_d1 = sess.run([model.loss,
            model.loss_rec, model.loss_adv, model.loss_d0, model.loss_d1],
            feed_dict=feed_dictionary(model, batch, args.rho, args.gamma_min))
        losses.add([loss, loss_rec, loss_adv, loss_d0, loss_d1])
    n0, n1 = len(data0), len(data1)
    data0_rec = reorder(order0, data0_rec)[:n0]
    data1_rec = reorder(order1, data1_rec)[:n1]

    data0_tsf = reorder(order0, data0_tsf)[:n0]
    data1_tsf = reorder(order1, data1_tsf)[:n1]
    if out_path:
        write_sent(data0_rec, out_path+'.0'+'.rec')
        write_sent(data1_rec, out_path+'.1'+'.rec')
        write_sent(data0_tsf, out_path+'.0'+'.tsf')
        write_sent(data1_tsf, out_path+'.1'+'.tsf')
    return losses

def create_model(sess, args, vocab):
    model = Model(args, vocab)
    if args.load_model:
        print ('Loading model from', args.model)
        model.saver.restore(sess, args.model)
    else:
        print ('Creating model with fresh parameters.')
        sess.run(tf.global_variables_initializer())
    
    return model

if __name__ == '__main__':
    tf.reset_default_graph() # in tu code asli nist
    args = load_arguments()

    #####   data preparation   #####
    if args.train:
        train0 = load_sent(args.train + '.0', args.max_train_size)
        train1 = load_sent(args.train + '.1', args.max_train_size)
        print ('#sents of training file 0:', len(train0))
        print ('#sents of training file 1:', len(train1))

        if not os.path.isfile(args.vocab):
            build_vocab(train0 + train1, args.vocab)

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print ('vocabulary size:', vocab.size)

    if args.dev:
        dev0 = load_sent(args.dev + '.0')
        dev1 = load_sent(args.dev + '.1')

    if args.test:
        test0 = load_sent(args.test + '.0')
        test1 = load_sent(args.test + '.1')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, args, vocab)

        if args.beam > 1:
            decoder = generation_method.BeamDecoder(sess, args, vocab, model)
        else:
            decoder = generation_method.GreedyDecoder(sess, args, vocab, model)

        if args.train:
            batches, _, _ = get_batches(train0, train1, vocab.word2id,
                args.batch_size,args, noisy=True)
            random.shuffle(batches)
            start_time = time.time()
            step = 0
            losses = Accumulator(args.steps_per_checkpoint,
                ['loss', 'rec', 'adv', 'd0', 'd1'])
            best_dev = float('inf')
            learning_rate = args.learning_rate
            rho = args.rho
            gamma = args.gamma_init
            dropout = args.dropout_keep_prob

            for epoch in range(1, 1+args.max_epochs):
                print( '--------------------epoch %d--------------------' % epoch)
                print ('learning_rate:', learning_rate, '  gamma:', gamma)

                for batch in batches:
                    feed_dict = feed_dictionary(model, batch, rho, gamma,
                        dropout, learning_rate)

                    loss_d0, _ = sess.run([model.loss_d0, model.optimize_d0],
                        feed_dict=feed_dict)
                    loss_d1, _ = sess.run([model.loss_d1, model.optimize_d1],
                        feed_dict=feed_dict)

                    # do not back-propagate from the discriminator when it functions poorly
                    if loss_d0 < 1.2 and loss_d1 < 1.2:
                        optimize0 = model.optimize_tot0
                        optimize1 = model.optimize_tot1
                    else:
                        optimize0 = model.optimize_rec0
                        optimize1 = model.optimize_rec1
                    loss, loss_rec, loss_adv, _ ,_= sess.run([model.loss,
                        model.loss_rec, model.loss_adv, optimize0, optimize1],
                        feed_dict=feed_dict)
                    losses.add([loss, loss_rec, loss_adv, loss_d0, loss_d1])

                    step += 1
                    if step % args.steps_per_checkpoint == 0:
                        losses.output('step %d, time %.0fs,'
                            % (step, time.time() - start_time))
                        losses.clear()

                if args.dev:
                    dev_losses = transfer(model, decoder, sess, args, vocab,
                        dev0, dev1, args.output + '.epoch%d' % epoch)
                    dev_losses.output('dev')
                    if dev_losses.values[0] < best_dev:
                        best_dev = dev_losses.values[0]
                        print ('saving model...')
                        model.saver.save(sess, args.model)

                gamma = max(args.gamma_min, gamma * args.gamma_decay)

        if args.test:
            print ('creating test loss')
            test_losses = transfer(model, decoder, sess, args, vocab,
                test0, test1, args.output)
            test_losses.output('test')
            values, names = test_losses.output_list()
            with open(args.output, 'w') as f:
                for n,v in zip(names, values):
                    f.write(str(n))
                    f.write('\t')
                    f.write(str(v))
                    f.write('\n')
            test_losses.output('test')

        if args.online_testing:
            while True:
                sys.stdout.write('> ')
                sys.stdout.flush()
                inp = sys.stdin.readline().rstrip()
                if inp == 'quit' or inp == 'exit':
                    break
                inp = inp.split()
                y = int(inp[0])
                sent = inp[1:]

                batch = get_batch([sent], [y], vocab.word2id)
                ori, tsf = decoder.rewrite(batch)
                print ('original:', ' '.join(w for w in ori[0]))
                print ('transfer:', ' '.join(w for w in tsf[0]))
