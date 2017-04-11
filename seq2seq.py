import tensorflow as tf
import numpy as np
import json
from lstm import lstm_cell, linear

meta = json.load(open('data/meta.json'))
w2idx, idx2w = meta['w2idx'], meta['idx2w']
vocab_size = len(w2idx)
encoder_unrollings, decoder_unrollings  = 20, 22
embedding_size = 256
mem_size = 512
batch_size = 16

question = tf.placeholder(tf.int32, shape=[batch_size, encoder_unrollings])
answer = tf.placeholder(tf.int32, shape=[batch_size, decoder_unrollings])
answer_x = answer[:, :-1]
answer_y = tf.reshape(answer[:, 1:], [-1])

embedded = tf.get_variable(name='embedded', shape=[vocab_size, embedding_size], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

def encoder(cx):
  outputs = []
  with tf.variable_scope('encoder') as scope:
    x = tf.nn.embedding_lookup(embedded, cx) # batch_size x unrolling x embedding_size
    mem = tf.constant(0.0, shape=[batch_size, mem_size], name='initial_memory')
    state = tf.constant(0.0, shape=[batch_size, mem_size], name='initial_state')

    for i in range(encoder_unrollings):
      mem, state = lstm_cell(x[:, i, :], mem, state, embedding_size, mem_size, 'lstm')
      scope.reuse_variables()
      outputs.append(mem)

  return outputs, state

def decoder(cx, mem, state):
  outputs = []
  with tf.variable_scope('decoder') as scope:
    x = tf.nn.embedding_lookup(embedded, cx) 
    
    for i in range(decoder_unrollings - 1):
      mem, state = lstm_cell(x[:, i, :], mem, state, embedding_size, mem_size, 'lstm')
      scope.reuse_variables()
      outputs.append(mem)

  return outputs, state

mems_encoder, state_encoder = encoder(question)
mems_decoder, _ = decoder(answer_x, mems_encoder[-1], state_encoder)
logits = tf.concat(values=mems_decoder, axis=0)
ys = linear(logits, shape=[mem_size, vocab_size], activation_fn=tf.identity, scope='linear')
total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ys, labels=answer_y))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(total_loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

idx_q = np.load('data/questions.npy')
idx_a = np.load('data/answers.npy')
print idx_q.shape, idx_a.shape
for i in xrange(1, 1000):
  for batch in xrange(len(idx_q)/batch_size):
    batch_qs = idx_q[batch*batch_size:(batch+1)*batch_size]
    batch_as = idx_a[batch*batch_size:(batch+1)*batch_size]

    _, loss = sess.run([optimizer, total_loss], feed_dict={question:batch_qs, answer:batch_as})
    print loss
