import tensorflow as tf
import numpy as np
import json
from lstm import lstm_cell

questions = np.load('data/questions.npy')
answers = np.load('data/answers.npy')
meta = json.load(open('data/meta.json'))
w2idx, idx2w = meta['w2idx'], meta['idx2w']
vocab_size = len(w2idx)
encoder_unrollings, decoder_unrollings  = 20, 22
embedding_size = 256
mem_size = 512
batch_size = 16

question = tf.placeholder(tf.int32, shape=[None, encoder_unrollings])
answer = tf.placeholder(tf.int32, shape=[None, decoder_unrollings])

def encoder():
  outputs = []
  with tf.variable_scope('encoder'):

    embedded = tf.get_variable(name='embedded', shape=[vocab_size, embedding_size], initializer=tf.truncated_normal(mean=0.0, stddev=0.1))
    x = tf.nn.embedding_lookup(embedded, question) # batch_size x unrolling x embedding_size
    mem = tf.constant(0, shape=[tf.shape(x)[0], mem_size], name='initial memory')
    state = tf.constant(0, shape=[tf.shape(x)[0], mem_size], name='initial state')

    for i in range(encoder_unrollings):
      mem, state = lstm_cell(x[:, i,:], mem, state, vocab_size, mem_size, 'lstm')
      outputs.append(output)

  return outputs, state

def decoder(mem, state):
  with tf.variable_scope('decoder'):
    embedded = tf.get_variable(name='embedded', shape=)
      
    


