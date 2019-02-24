import tensorflow as tf
import numpy as np
import math

def rbf_kernel(Q, K, gamma=0.5):
  Q = tf.transpose(Q, [1, 0, 2]) # Time major.
  K = tf.transpose(K, [1, 0, 2])
  return tf.transpose(tf.map_fn(
    lambda k: tf.exp(-gamma * tf.reduce_sum(tf.square(Q - k), axis=-1)),
    K
  ), [2, 1, 0])

def bahdanau(Q, K):
  Q = tf.transpose(Q, [1, 0, 2]) # Time major.
  K = tf.transpose(K, [1, 0, 2])
  attention = tf.map_fn(
    lambda k: tf.layers.dense(tf.tanh(Q + k), 1), 
    K
  )
  return tf.transpose(tf.squeeze(attention, axis=-1), [2, 1, 0]) 

def exact_match(Q, K):
  Q = tf.transpose(Q, [1, 0, 2]) # Time major.
  K = tf.transpose(K, [1, 0, 2])

  attention = tf.map_fn(
    lambda k: tf.cast(tf.reduce_all(tf.equal(Q, k), axis=-1), tf.int64),
    K
  )

  return tf.cast(tf.transpose(attention, [2, 1, 0]), tf.float32)

def dot_product(Q, K, scaled=False, cosine=False):
  attention = tf.matmul(Q, K, transpose_b=True)

  if scaled:
    attention = tf.divide(attention, math.sqrt(Q.shape[2].value))

  if cosine:
    norm_q = tf.sqrt(tf.reduce_sum(tf.square(Q), axis=-1, keepdims=True))
    norm_k = tf.sqrt(tf.reduce_sum(tf.square(K), axis=-1, keepdims=True))
    norm = tf.matmul(norm_q, norm_k, transpose_b=True)
    attention = tf.abs(tf.divide(attention, norm))

  return attention

def normalize(inputs, epsilon = 1e-8):
  inputs_shape = inputs.get_shape()
  params_shape = inputs_shape[-1:]
  
  mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
  beta= tf.Variable(tf.zeros(params_shape))
  gamma = tf.Variable(tf.ones(params_shape))
  normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
  return gamma * normalized + beta

def pos_embeddings(inputs, emb_dim, max_length=2000):
  position_emb = np.array([
    [(pos+1) / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
    for pos in range(max_length)
  ])
  
  position_emb[:,0::2] = np.sin(position_emb[:,0::2]) # dim 2i
  position_emb[:,1::2] = np.cos(position_emb[:,1::2]) # dim 2i+1

  N = tf.shape(inputs)[0]
  T = tf.shape(inputs)[1]
  lookup_table = tf.convert_to_tensor(position_emb, dtype=tf.float32)

  position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
  pos_embeddings = tf.nn.embedding_lookup(lookup_table, position_ind)
  return inputs + pos_embeddings

def attention(queries, keys, num_heads, values=None, residual='add', use_pos_embeddings=True, queries_eq_keys=False, training=False):
  # attention_size = queries.shape[2].value
  attention_size = 100

  if values is None:
    values = keys

  output_size = values.shape[2].value

  Q = tf.layers.dense(queries, attention_size)

  if use_pos_embeddings:
    Q = pos_embeddings(Q, attention_size)

  if queries_eq_keys:
    K = Q
  else:
    K = tf.layers.dense(keys, attention_size)
  V = tf.layers.dense(values, output_size)

  # Split and concat
  Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T, H/h) 
  K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T, H/h) 
  V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T, H/h) 

  # Similarity function.
  attention = dot_product(Q_, K_, scaled=True)

  # Key Masking.
  # key_masks = tf.sign(tf.reduce_sum(tf.abs(K), axis=-1))
  # key_masks = tf.tile(key_masks, [num_heads, 1])
  # key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(Q)[1], 1])
  # paddings = tf.ones_like(attention)*(-2**32+1)
  # attention = tf.where(tf.equal(key_masks, 0), paddings, attention)

  # Regularization.
  alphas = tf.nn.softmax(attention, name='alphas')

  # Query Masking
  # query_masks = tf.sign(tf.reduce_sum(tf.abs(Q), axis=-1))
  # query_masks = tf.tile(query_masks, [num_heads, 1])
  # query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(K)[1]])
  # alphas *= query_masks

  alphas = tf.layers.dropout(alphas, rate=0.5, training=training)
  outputs = tf.matmul(alphas, V_)
  outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

  if residual == 'add':
    outputs += values
  elif residual == 'concat':
    outputs = tf.concat([outputs, values], axis=-1)

  # Layer normalization.
  # outputs = normalize(outputs)
  return outputs

def exact_attention(queries, keys, values, residual='add', training=False):
  output_size = values.shape[2].value
  Q = queries
  K = keys
  V = tf.layers.dense(values, output_size)

  # Similarity function.
  attention = exact_match(Q, K)

  # Key Masking.
  # key_masks = tf.sign(tf.reduce_sum(tf.abs(K), axis=-1))
  # key_masks = tf.tile(key_masks, [1, 1])
  # key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(Q)[1], 1])
  # paddings = tf.ones_like(attention)*(-2**32+1)
  # attention = tf.where(tf.equal(key_masks, 0), paddings, attention)

  # Regularization.
  alphas = tf.nn.softmax(attention, name='alphas')

  # Query Masking
  # query_masks = tf.sign(tf.reduce_sum(tf.abs(Q), axis=-1))
  # query_masks = tf.tile(query_masks, [1, 1])
  # query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(K)[1]])
  # alphas *= query_masks

  alphas = tf.layers.dropout(alphas, rate=0.5, training=training)
  outputs = tf.matmul(alphas, V)

  if residual == 'add':
    outputs += values
  elif residual == 'concat':
    outputs = tf.concat([outputs, values], axis=-1)

  return outputs


