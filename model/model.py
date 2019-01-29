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
    'dim_chars': 100,
    # 'dim_chars': 300,
    'char_lstm_size': 25,
    'dim': 300,
    'dropout': 0.5,
    'num_oov_buckets': 1,
    'filters': 50,
    'kernel_size': 3,
    'lstm_size': 400,
    'learning_rate': 0.001,
    'words': str(Path(DATADIR, 'vocab.words.txt')),
    'chars': str(Path(DATADIR, 'vocab.chars.txt')),
    'tags': str(Path(DATADIR, 'vocab.tags.txt')),
    'glove': str(Path(DATADIR, 'glove.npz')),
    'char_embeddings': str(Path(DATADIR, 'char_embeddings.npz')),
    'similarity_fn': 'scaled_dot',
    'regularization_fn': 'softmax',
    'use_attention': True
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

def dot_product(Q, K, scaled=False, cosine=False):
  attention = tf.matmul(Q, K, transpose_b=True)

  if scaled:
    attention = tf.divide(attention, math.sqrt(Q.shape[2].value))

  if cosine:
    norm_q = tf.sqrt(tf.reduce_sum(tf.square(Q), axis=-1, keepdims=True))
    norm_k = tf.sqrt(tf.reduce_sum(tf.square(K), axis=-1, keepdims=True))
    norm = tf.matmul(norm_q, norm_k, transpose_b=True)
    attention = tf.abs(tf.divide(alphas, norm))

  return attention

def self_attention(inputs, outputs, name='shsa'):
  attention_size = inputs.shape[2].value
  output_size = outputs.shape[2].value

  # Q = tf.layers.dense(inputs, attention_size)
  # K = tf.layers.dense(inputs, attention_size)
  # V = tf.layers.dense(inputs, output_size)
  Q = inputs
  K = inputs
  V = inputs

  # Similarity function.
  if   params['similarity_fn'] == 'rbf':
    attention = rbf_kernel(Q, K)
  elif params['similarity_fn'] == 'dot':
    attention = dot_product(Q, K)
  elif params['similarity_fn'] == 'scaled_dot':
    attention = dot_product(Q, K, scaled=True)
  elif params['similarity_fn'] == 'cosine':
    attention = dot_product(Q, K, cosine=True)
  elif params['similarity_fn'] == 'bahdanau':
    attention = bahdanau(Q, K)

  # Regularization.
  if   params['regularization_fn'] == 'softmax':
    alphas = tf.nn.softmax(attention, name=name)
  elif params['regularization_fn'] == 'tanh':
    alphas = tf.tanh(attention, name=name)
  elif params['regularization_fn'] == 'linear':
    regularizer = tf.reduce_sum(attention, keepdims=True, axis=-1)
    alphas = tf.divide(attention, regularizer, name=name)

  return tf.matmul(alphas, V)

def multi_head_self_attention(inputs, outputs):
  hidden_size = inputs.shape[2].value
  output_size = outputs.shape[2].value

  num_heads = 8
  att_layers = []
  for i in range(num_heads):
    att_layers.append(single_head_self_attention(inputs, outputs, name='alphas' + str(i)))

  return tf.layers.dense(tf.concat(att_layers, axis=-1), output_size)

def attention(inputs, outputs, training=False):
  return self_attention(inputs, outputs, name='alphas0')

def position_embeddings(max_length, emb_dim, inputs):
  position_emb = np.array([
      [(pos+1) / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
      for pos in range(max_length)
  ])
  
  position_emb[:,0::2] = np.sin(position_emb[:,0::2]) # dim 2i
  position_emb[:,1::2] = np.cos(position_emb[:,1::2]) # dim 2i+1

  variable = np.vstack([position_emb, [[0.] * emb_dim]])
  variable = tf.Variable(variable, dtype=tf.float32, trainable=False)

  # variable = position_embeddings(5000, params['lstm_size'])
  pos = tf.slice(tf.constant(np.arange(100000), dtype=tf.int32), [0], [tf.shape(inputs)[1]])
  pos_embeddings = tf.nn.embedding_lookup(variable, pos)
  return inputs + pos_embeddings

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
    # pretrained_chars = np.load(params['char_embeddings'])['embeddings']  # np.array
    # variable = np.vstack([pretrained_chars, [[0.] * params['dim_chars']]])
    # variable = tf.Variable(variable, dtype=tf.float32, trainable=True)
    variable = tf.get_variable('chars_embeddings', [num_chars + 1, params['dim_chars']], tf.float32)
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
    word_embeddings = position_embeddings(5000, params['dim'], word_embeddings)
  
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
    output = tf.layers.dropout(output, rate=0.5, training=training)

    # Position embeddings.
    # output = position_embeddings(5000, params['lstm_size'], output)
  
    output2 = attention(output, output, training=training)
    output = output + output2
    logits = tf.layers.dense(output, num_tags)
    # logits = logits + att_output

    # logits = tf.nn.relu(logits + output2)
    # output = tf.concat([output, output2], axis=-1)
  
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
