import tensorflow as tf

def lstm_cell(i, m, state, embedding_size, mem_size, scope):
  with tf.variable_scope(scope):
    ifcox = tf.get_variable(name='ifcox', shape=[embedding_size, 4*mem_size],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    ifcom = tf.get_variable(name='ifcom', shape=[mem_size, 4*mem_size],  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    ifcob = tf.get_variable(name='ifcob', shape=[4*mem_size], initializer=tf.constant_initializer(0))
  
    all_gates = tf.matmul(i, ifcox) + tf.matmul(m, ifcom) + ifcob
    in_gate = tf.sigmoid(all_gates[:, :mem_size])
    forget_gate = tf.sigmoid(all_gates[:, mem_size:2*mem_size])
    c = tf.tanh(all_gates[:, 2*mem_size:3*mem_size])
    out_gate = tf.sigmoid(all_gates[:, 3*mem_size:])

    state = forget_gate*state + in_gate*c
    
    return out_gate*tf.tanh(state), state

def linear(x, shape, activation_fn, scope, reuse):
  with tf.variable_scope(scope, reuse=reuse):
    W = tf.get_variable(name='W', shape=shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    b = tf.get_variable(name='b', shape=shape[-1], initializer=tf.constant_initializer(0))
    h = tf.matmul(x, W) + b
    a = activation_fn(h)

  return a
