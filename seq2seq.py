import tensorflow as tf
import numpy as np
import json
import sys
sys.path.append('data/')
import utils
from lstm import lstm_cell, linear

meta = json.load(open('data/meta.json'))
w2idx, idx2w = meta['w2idx'], meta['idx2w']
vocab_size = len(w2idx)
PAD = w2idx['_']

encoder_unrollings, decoder_unrollings  = 20, 22
embedding_size = 256
mem_size = 512
batch_size = 256


question = tf.placeholder(tf.int32, shape=[batch_size, encoder_unrollings])
answer = tf.placeholder(tf.int32, shape=[batch_size, decoder_unrollings])
answer_x = answer[:, :-1]
answer_y = tf.reshape(answer[:, 1:], [-1])

embedded = tf.get_variable(name='embedded', shape=[vocab_size, embedding_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

def encoder(cx, unrollings, encoder_reuse):
  outputs = []
  with tf.variable_scope('encoder', reuse=encoder_reuse) as scope:
    x = tf.nn.embedding_lookup(embedded, cx) # batch_size x unrolling x embedding_size
    mem = tf.constant(0.0, shape=[batch_size, mem_size], name='initial_memory')
    state = tf.constant(0.0, shape=[batch_size, mem_size], name='initial_state')

    for i in range(unrollings):
      mem, state = lstm_cell(x[:, i, :], mem, state, embedding_size, mem_size, 'lstm')
      scope.reuse_variables()
      outputs.append(mem)

  return outputs, state

def decoder(cx, mem, state, unrollings, decoder_reuse):
  outputs = []
  with tf.variable_scope('decoder', reuse=decoder_reuse) as scope:
    x = tf.nn.embedding_lookup(embedded, cx) 
    
    for i in range(unrollings):
      mem, state = lstm_cell(x[:, i, :], mem, state, embedding_size, mem_size, 'lstm')
      scope.reuse_variables()
      outputs.append(mem)

  return outputs, state

def max_sampling(max_length):
  ys = []
  x = tf.constant(w2idx['BOS'], shape=[batch_size, 1])

  #with tf.variable_scope('max_sampling', reuse=True):  
  mems_encoder, state_encoder = encoder(question, encoder_unrollings, encoder_reuse=True)

  # assign last mem and state to decoder net
  mem_decoder = [mems_encoder[-1]]
  state_decoder = state_encoder
  for _ in range(max_length):
    mem_decoder, state_decoder = decoder(x, mem_decoder[-1], state_decoder, 1, decoder_reuse=True)
    _y = linear(mem_decoder[0], shape=[mem_size, vocab_size], activation_fn=tf.identity, scope='linear', reuse=True)
    y = tf.argmax(_y, axis=1)

    x = tf.reshape(y, [-1, 1])
    ys.append(y)

  return ys

mems_encoder, state_encoder = encoder(question, encoder_unrollings, encoder_reuse=False)
mems_decoder, _ = decoder(answer_x, mems_encoder[-1], state_encoder, decoder_unrollings - 1, decoder_reuse=False)
logits = tf.concat(values=mems_decoder, axis=0)
ys = linear(logits, shape=[mem_size, vocab_size], activation_fn=tf.identity, scope='linear', reuse=False)

mask = tf.cast(tf.not_equal(answer_y, PAD), tf.float32)
total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ys, labels=answer_y)*mask
total_loss = tf.reduce_mean(total_loss, name='total_loss')

sampling = max_sampling(20)

var =  tf.get_collection(tf.GraphKeys.VARIABLES)
print [v.name for v in var], len(var)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(total_loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())



###########
summary_writer = tf.summary.FileWriter('logs/',graph_def=sess.graph_def)

#######
idx_q = np.load('data/questions.npy')
idx_a = np.load('data/answers.npy')
idx_q_sample = idx_q[:batch_size]
for q in idx_q_sample:
  print utils.idxs2str(q, idx2w)

print idx_q.shape, idx_a.shape
n = 0
for i in xrange(1, 1000):
  for batch in xrange(len(idx_q)/batch_size):
    batch_qs = idx_q[batch*batch_size:(batch+1)*batch_size]
    batch_as = idx_a[batch*batch_size:(batch+1)*batch_size]
    
    _, loss = sess.run([optimizer, total_loss], feed_dict={question:batch_qs, answer:batch_as})
    if n % 100 == 0:
      ys_sampling = sess.run(sampling, feed_dict={question: idx_q_sample}) 
      for y_sampling in np.transpose(ys_sampling):
        a_sampling = utils.idxs2str(y_sampling, idx2w)
        print a_sampling, '\n'

      print 'Iter', n, ': ', loss
    
    n += 1
