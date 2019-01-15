import numpy as np 
import tensorflow as tf
from pathlib import Path
from model.cnn import masked_conv1d_and_max

MAX_TOKEN_LENGTH = 10
MINIBATCH_SIZE = 10
DATADIR = 'data/conll2003'
LABEL_COL = 3

# Params
params = {
    'dim_chars': 100,
    'dim': 300,
    'dropout': 0.5,
    'num_oov_buckets': 1,
    'epochs': 25,
    'batch_size': 20,
    'buffer': 15000,
    'filters': 50,
    'kernel_size': 3,
    'lstm_size': 100,
    'words': str(Path(DATADIR, 'vocab.words.txt')),
    'chars': str(Path(DATADIR, 'vocab.chars.txt')),
    'tags': str(Path(DATADIR, 'vocab.tags.txt')),
    'glove': str(Path(DATADIR, 'glove.npz')),
    'fulldoc': False
}

# def bahdanau_attention(inputs, attention_size):
#   pass
# 
# def luong_attention(inputs, attention_size):
#   hidden_size = inputs.shape[2].value
#   print(inputs)
# 
#   # Dense layer.
#   w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
#   b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
#   print(w_omega)
#   v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega, name='v')
#   print(v)
# 
#   u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
#   vu = tf.tensordot(v, u_omega, axes=1, name='vu')
#   print(vu)
#   
#   alphas = tf.nn.softmax(vu, name='alphas')
#   print(tf.expand_dims(alphas, -1))
# 
#   output = inputs * tf.expand_dims(alphas, -1)
#   return output, alphas

def john_attention(inputs, attention_size):
  hidden_size = inputs.shape[2].value

  # Dense layer.
  w1 = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
  b1 = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

  w2 = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
  b2 = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

  u = tf.tanh(tf.tensordot(inputs, w1, axes=1) + b1, name='u')
  v = tf.tanh(tf.tensordot(inputs, w2, axes=1) + b2, name='v')
 
  v = tf.transpose(v, [0, 2, 1])
  m = tf.matmul(u, v)
  alphas = tf.nn.softmax(m, name='alphas')
  outputs = tf.matmul(alphas, inputs)

  return outputs, alphas

def attention(inputs, attention_size):
  return john_attention(inputs, attention_size)

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
    variable = tf.get_variable(
      'chars_embeddings', [num_chars + 1, params['dim_chars']], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(variable, char_ids)
    char_embeddings = tf.layers.dropout(char_embeddings, rate=dropout, training=training)
  
    # Char 1d convolution
    weights = tf.sequence_mask(nchars)
    char_embeddings = masked_conv1d_and_max(char_embeddings, weights, params['filters'], params['kernel_size'])
  
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
  
    output = tf.concat([output_fw, output_bw], axis=-1)
    output, _ = attention(output, 100)
    output = tf.layers.dropout(output, rate=dropout, training=training)
  
  with tf.name_scope('output'):
    logits = tf.layers.dense(output, num_tags)
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
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
  
  with tf.name_scope('prediction'):
    reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
      params['tags'])
    pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))  
