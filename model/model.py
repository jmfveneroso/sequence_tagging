import numpy as np 
import tensorflow as tf
from pathlib import Path
from model.cnn import masked_conv1d_and_max
import math

MAX_TOKEN_LENGTH = 10
MINIBATCH_SIZE = 10
DATADIR = 'data/conll2003'
LABEL_COL = 3

# Params
params = {
    # 'dim_chars': 100,
    'dim_chars': 300,
    'char_lstm_size': 25,
    'dim': 300,
    'dropout': 0.5,
    'num_oov_buckets': 1,
    'filters': 50,
    'kernel_size': 3,
    'lstm_size': 200,
    'attention_size': 150,
    'learning_rate': 0.001,
    'words': str(Path(DATADIR, 'vocab.words.txt')),
    'chars': str(Path(DATADIR, 'vocab.chars.txt')),
    'tags': str(Path(DATADIR, 'vocab.tags.txt')),
    'glove': str(Path(DATADIR, 'glove.npz')),
    'char_embeddings': str(Path(DATADIR, 'char_embeddings.npz'))
}

def lstm_char_representations(char_embeddings, nchars):
  with tf.variable_scope('char_embeddings'):
    dim_words = tf.shape(char_embeddings)[1]
    dim_chars = tf.shape(char_embeddings)[2]
    t = tf.reshape(char_embeddings, [-1, dim_chars, params['dim_chars']])

    lstm_cell_fw_c = tf.nn.rnn_cell.LSTMCell(params['char_lstm_size'])
    lstm_cell_bw_c = tf.nn.rnn_cell.LSTMCell(params['char_lstm_size'])
    
    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
      lstm_cell_fw_c, lstm_cell_bw_c, t,
      dtype=tf.float32,
      sequence_length=tf.reshape(nchars, [-1]),
    )

    output = tf.concat([output_fw, output_bw], 2)
    output = output[:, -1, :] 
    char_embeddings = tf.reshape(output, [-1, dim_words, 50])
  return char_embeddings

def dot_product(x, name='dot'):
  return tf.matmul(x, tf.transpose(x, [0, 2, 1]), name=name)

def norm_dot_product(x, name='norm_dot'):
  alphas = dot_product(x)
  l = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
  l = tf.matmul(l, tf.transpose(l, [0, 2, 1]))
  alphas = tf.divide(alphas, l)
  alphas = tf.abs(alphas, name=name)
  return alphas

def rbf_kernel(x, sigma=0.5, name='rbf'):
  def fn(xi):
    return tf.exp(-tf.reduce_sum(tf.square(x - xi), axis=-1) * sigma)
  
  x = tf.transpose(x, [1, 0, 2]) # Time major.
  z = tf.map_fn(fn, x)
  return tf.transpose(z, [2, 1, 0], name=name) 

def bahdanau(x, attention_size=100, name='bahdanau'):
  hidden_size = x.shape[2].value

  w_v = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
  b_v = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

  w_u = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
  b_u = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

  w_z = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
  b_z = tf.Variable(tf.random_normal([], stddev=0.1))

  u = tf.tensordot(x, w_u, axes=1) + b_u
  v = tf.tensordot(x, w_v, axes=1) + b_v
  def fn(vi):
    aux = tf.tanh(u + vi)
    return tf.tensordot(aux, w_z, axes=1) + b_z
  
  u = tf.transpose(u, [1, 0, 2]) # Time major.
  v = tf.transpose(v, [1, 0, 2]) # Time major.
  z = tf.map_fn(fn, v)
  return tf.transpose(z, [2, 1, 0], name=name) 

def single_head_self_attention(inputs, outputs, name='shsa'):
  hidden_size = inputs.shape[2].value

  queries = tf.layers.dense(inputs, hidden_size)
  keys    = queries
  values  = tf.layers.dense(outputs, hidden_size)

  dot = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))
  scaled_dot = tf.divide(dot, math.sqrt(hidden_size))

  alphas = tf.nn.softmax(scaled_dot, name=name)
  return tf.matmul(alphas, values)

def multi_head_self_attention(inputs, outputs):
  hidden_size = inputs.shape[2].value

  num_heads = 2
  att_layers = []
  for i in range(num_heads):
    att_layers.append(single_head_self_attention(inputs, outputs, name='alphas' + str(i)))

  return tf.layers.dense(tf.concat(att_layers, axis=-1), hidden_size)

def attention(inputs, outputs, attention_size, training=False):
  return multi_head_self_attention(inputs, outputs)
  # return self_attention(input1, input2, training=training)

  # hidden_size = input1.shape[2].value

  # w1 = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
  # b1 = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
  # u = tf.tensordot(input1, w1, axes=1) + b1
 
  # 

  # # u = tf.layers.dropout(u, rate=params['dropout'], training=training)
  # # u = input1

  # # Second feed forward layer.
  # # w2 = tf.Variable(tf.random_normal([attention_size, attention_size], stddev=0.1))
  # # b2 = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
  # # u = tf.tensordot(u, w2, axes=1) + b2

  # # sigma = 1.0 / (math.sqrt(attention_size))
  # # sigma = 1.0
  # # alphas = rbf_kernel(u, sigma=sigma, name='palphas')
  # alphas = norm_dot_product(u, name='palphas')
  # # alphas = dot_product(u, name='palphas')
  # # alphas = bahdanau(u)

  # # Regularization
  # alphas = tf.nn.softmax(alphas, name='alphas')
  # # alphas = tf.divide(alphas, tf.reduce_sum(alphas, keepdims=True, axis=-1) , name='alphas')
  # # alphas = tf.tanh(m, name='alphas')

  # # w = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1))
  # # b = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))

  # # alphas = tf.nn.softmax(alphas, name='alphas')
  # output = tf.matmul(alphas, input2, name='weighted_lstm_states')
 
  # return output 

def create_model():
  with Path(params['tags']).open() as f:
    indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
    num_tags = len(indices) + 1
  
  with Path(params['chars']).open() as f:
    num_chars = sum(1 for _ in f) + params['num_oov_buckets']
  
  with tf.name_scope('inputs'):
    words = tf.placeholder(tf.string, shape=(None, None), name='words')
    nwords = tf.placeholder(tf.int32, shape=(None,), name='nwords')
    chars = tf.placeholder(tf.string, shape=(None, None, None), name='chars')
    nchars = tf.placeholder(tf.int32, shape=(None, None), name='nchars')
    labels = tf.placeholder(tf.string, shape=(None, None), name='labels')
    training = tf.placeholder(tf.bool, shape=(), name='training')
  
  with tf.name_scope('embeddings'):
    dropout = params['dropout']
    vocab_words = tf.contrib.lookup.index_table_from_file(
      params['words'], num_oov_buckets=params['num_oov_buckets']
    )
    vocab_chars = tf.contrib.lookup.index_table_from_file(
      params['chars'], num_oov_buckets=params['num_oov_buckets']
    )
  
    # Char Embeddings
    char_ids = vocab_chars.lookup(chars)
    pretrained_chars = np.load(params['char_embeddings'])['embeddings']  # np.array
    variable = np.vstack([pretrained_chars, [[0.] * params['dim_chars']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=True)
    # variable = tf.get_variable('chars_embeddings', [num_chars + 1, params['dim_chars']], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
    char_embeddings_pre = tf.layers.dropout(char_embeddings, rate=dropout, training=training)
  
    # Char 1d convolution
    weights = tf.sequence_mask(nchars)
    # char_embeddings = masked_conv1d_and_max(char_embeddings_pre, weights, params['filters'], params['kernel_size']) 
    char_embeddings = lstm_char_representations(char_embeddings, nchars)

    # Word Embeddings
    word_ids = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']  # np.array
    variable = np.vstack([glove, [[0.] * params['dim']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(variable, word_ids)
  
    # Concatenate Word and Char Embeddings
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)
  
  with tf.name_scope('lstm'):
    t = embeddings
  
    lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(params['lstm_size'])
    lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(params['lstm_size'])
  
    (output_fw, output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(
      lstm_cell_fw, lstm_cell_bw, t,
      dtype=tf.float32,
      sequence_length=nwords
    )

    # output = tf.concat([output_fw, output_bw], axis=-1, name='lstm_states')
    output = tf.add(output_fw, output_bw, name='lstm_states')
    output = tf.layers.dropout(output, rate=dropout, training=training)

    output2 = attention(t, output, params['attention_size'], training=training)
    
    # output = tf.nn.relu(output + output2)
    output = tf.concat([output, output2], axis=-1)

    # output = tf.concat([output, output2], axis=-1)
    logits = tf.layers.dense(output, num_tags)
  
  with tf.name_scope('output'):
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)
  
  with tf.name_scope('loss'):
    vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
    tags = vocab_tags.lookup(labels)
    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
      logits, tags, nwords, crf_params)
    loss = tf.reduce_mean(-log_likelihood, name='loss')
  
    correct = tf.equal(tf.to_int64(pred_ids), tags)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
  
  with tf.name_scope('training'):
    train_step = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(loss)
  
  with tf.name_scope('prediction'):
    reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
      params['tags'])
    pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))  
