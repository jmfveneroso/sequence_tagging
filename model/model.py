import math
import numpy as np 
import tensorflow as tf
from pathlib import Path
from six.moves import reduce
import tensorflow_hub as hub

class SequenceModel:
  def __init__(self, params=None):
    DATADIR = 'data/conll2003'
    self.params = {
      'words': str(Path(DATADIR, 'vocab.words.txt')),
      'chars': str(Path(DATADIR, 'vocab.chars.txt')),
      'tags': str(Path(DATADIR, 'vocab.tags.txt')),
      'glove': str(Path(DATADIR, 'glove.npz')),
      'char_embeddings': str(Path(DATADIR, 'char_embeddings.npz')),
      'learning_rate': 0.001,
      'dim_chars': 100,
      'dim_words': 300,
      'dropout': 0.5,
      'lstm_size': 200,
      'filters': 50,
      'kernel_size': 3,
      'char_lstm_size': 25,
      'pretrained_chars': False,
      'use_attention': True,
      'use_crf': True,
      'char_representation': 'cnn',
      'num_heads': 2,
      'similarity_fn': 'scaled_dot', # rbf, dot, scaled_dot, cosine, bahdanau
      'regularization_fn': 'softmax',
      'pos_embeddings': 'lstm',
      'queries_eq_keys': True,
      'residual': True,
      'elmo': False,
    }
    params = params if params is not None else {}
    self.params.update(params)

  def cnn(self, t, weights, filters, kernel_size):
    shape = tf.shape(t)
    ndims = t.shape.ndims
    dim1 = reduce(lambda x, y: x*y, [shape[i] for i in range(ndims - 2)])
    dim2 = shape[-2]
    dim3 = t.shape[-1]
  
    # Reshape weights
    weights = tf.reshape(weights, shape=[dim1, dim2, 1])
    weights = tf.to_float(weights)
  
    # Reshape input and apply weights
    flat_shape = [dim1, dim2, dim3]
    t = tf.reshape(t, shape=flat_shape)
    t *= weights
  
    # Apply convolution
    t_conv = tf.layers.conv1d(t, filters, kernel_size, padding='same')
    t_conv *= weights
  
    # Reduce max -- set to zero if all padded
    t_conv += (1. - weights) * tf.reduce_min(t_conv, axis=-2, keepdims=True)
    t_max = tf.reduce_max(t_conv, axis=-2)
  
    # Reshape the output
    final_shape = [shape[i] for i in range(ndims-2)] + [filters]
    t_max = tf.reshape(t_max, shape=final_shape)
    return t_max

  def lstm_char_representations(self, char_embeddings, nchars):
    with tf.variable_scope('lstm_chars'):
      dim_words = tf.shape(char_embeddings)[1]
      dim_chars = tf.shape(char_embeddings)[2]

      t = tf.reshape(char_embeddings, [-1, dim_chars, self.params['dim_chars']])
  
      lstm_cell_fw_c = tf.nn.rnn_cell.LSTMCell(self.params['char_lstm_size'])
      lstm_cell_bw_c = tf.nn.rnn_cell.LSTMCell(self.params['char_lstm_size'])
      
      (_, _), (output_fw, output_bw) = tf.nn.bidirectional_dynamic_rnn(
        lstm_cell_fw_c, lstm_cell_bw_c, t,
        dtype=tf.float32,
        sequence_length=tf.reshape(self.nchars, [-1]),
      )

      # output_fw[0] is the cell state and output_fw[1] is the hidden state.
      output = tf.concat([output_fw[1], output_bw[1]], axis=-1)
      return tf.reshape(output, [-1, dim_words, 2*self.params['char_lstm_size']])
  
  def rbf_kernel(self, Q, K, gamma=0.5):
    Q = tf.transpose(Q, [1, 0, 2]) # Time major.
    K = tf.transpose(K, [1, 0, 2])
    return tf.transpose(tf.map_fn(
      lambda k: tf.exp(-gamma * tf.reduce_sum(tf.square(Q - k), axis=-1)),
      K
    ), [2, 1, 0])

  def bahdanau(self, Q, K):
    Q = tf.transpose(Q, [1, 0, 2]) # Time major.
    K = tf.transpose(K, [1, 0, 2])
    attention = tf.map_fn(
      lambda k: tf.layers.dense(tf.tanh(Q + k), 1), 
      K
    )
    return tf.transpose(tf.squeeze(attention, axis=-1), [2, 1, 0]) 
  
  def dot_product(self, Q, K, scaled=False, cosine=False):
    attention = tf.matmul(Q, K, transpose_b=True)
  
    if scaled:
      attention = tf.divide(attention, math.sqrt(Q.shape[2].value))
  
    if cosine:
      norm_q = tf.sqrt(tf.reduce_sum(tf.square(Q), axis=-1, keepdims=True))
      norm_k = tf.sqrt(tf.reduce_sum(tf.square(K), axis=-1, keepdims=True))
      norm = tf.matmul(norm_q, norm_k, transpose_b=True)
      attention = tf.abs(tf.divide(attention, norm))
  
    return attention

  def normalize(self, inputs, epsilon = 1e-8):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    beta= tf.Variable(tf.zeros(params_shape))
    gamma = tf.Variable(tf.ones(params_shape))
    normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
    return gamma * normalized + beta
  
  def attention(self, inputs, outputs):
    attention_size = inputs.shape[2].value
    output_size = outputs.shape[2].value
  
    Q = tf.layers.dense(inputs, attention_size)
    if self.params['queries_eq_keys']:
      K = Q
    else:
      K = tf.layers.dense(inputs, attention_size)
    V = tf.layers.dense(inputs, output_size)

    # Split and concat
    num_heads = self.params['num_heads']
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T, H/h) 
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T, H/h) 
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T, H/h) 
  
    # Similarity function.
    if   self.params['similarity_fn'] == 'rbf':
      attention = self.rbf_kernel(Q_, K_)
    elif self.params['similarity_fn'] == 'dot':
      attention = self.dot_product(Q_, K_)
    elif self.params['similarity_fn'] == 'scaled_dot':
      attention = self.dot_product(Q_, K_, scaled=True)
    elif self.params['similarity_fn'] == 'cosine':
      attention = self.dot_product(Q_, K_, cosine=True)
    elif self.params['similarity_fn'] == 'bahdanau':
      attention = self.bahdanau(Q_, K_)

    # Key Masking.
    key_masks = tf.sign(tf.reduce_sum(tf.abs(K), axis=-1))
    key_masks = tf.tile(key_masks, [num_heads, 1])
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(Q)[1], 1])
    paddings = tf.ones_like(attention)*(-2**32+1)
    attention = tf.where(tf.equal(key_masks, 0), paddings, attention)
  
    # Regularization.
    name = 'alphas'
    if   self.params['regularization_fn'] == 'softmax':
      alphas = tf.nn.softmax(attention, name=name)
    elif self.params['regularization_fn'] == 'tanh':
      alphas = tf.tanh(attention, name=name)
    elif self.params['regularization_fn'] == 'linear':
      regularizer = tf.reduce_sum(attention, keepdims=True, axis=-1)
      alphas = tf.divide(attention, regularizer, name=name)

    # Query Masking
    query_masks = tf.sign(tf.reduce_sum(tf.abs(Q), axis=-1))
    query_masks = tf.tile(query_masks, [num_heads, 1])
    query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(K)[1]])
    outputs *= query_masks
  
    alphas = self.dropout(alphas)
    outputs = tf.matmul(alphas, V_)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

    if self.params['residual'] == 'add':
      outputs += inputs
    elif self.params['residual'] == 'concat':
      outputs = tf.concat([outputs, inputs], axis=-1)

    outputs = self.normalize(outputs)
    return outputs
  
  def position_embeddings(self, max_length, emb_dim, inputs):
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
  
  def dropout(self, x):
    return tf.layers.dropout(x, rate=self.params['dropout'], training=self.training)

  def create_placeholders(self):
    with tf.name_scope('inputs'):
      self.words    = tf.placeholder(tf.string, shape=(None, None),       name='words'   )
      self.nwords   = tf.placeholder(tf.int32,  shape=(None,),            name='nwords'  )
      self.uids     = tf.placeholder(tf.int32,  shape=(None,None),        name='uids'    )
      self.chars    = tf.placeholder(tf.string, shape=(None, None, None), name='chars'   )
      self.nchars   = tf.placeholder(tf.int32,  shape=(None, None),       name='nchars'  )
      self.labels   = tf.placeholder(tf.string, shape=(None, None),       name='labels'  )
      self.training = tf.placeholder(tf.bool,   shape=(),                 name='training')

  def elmo_embeddings(self):
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
    
    word_embeddings = elmo(
      inputs={
        "tokens": self.words,
        "sequence_len": self.nwords
      },
      signature="tokens",
      as_dict=True
    )["elmo"]

    if self.params['pos_embeddings'] == 'word':
      word_embeddings = self.position_embeddings(
        1600, 
        1024, 
        word_embeddings
      )

    return word_embeddings

  def word_embeddings(self):
    vocab_words = tf.contrib.lookup.index_table_from_file(
      self.params['words'], num_oov_buckets=1
    )

    word_ids = vocab_words.lookup(self.words)
    glove = np.load(self.params['glove'])['embeddings']
    variable = np.vstack([glove, [[0.] * self.params['dim_words']]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

    if self.params['pos_embeddings'] == 'word':
      word_embeddings = self.position_embeddings(
        1600, 
        self.params['dim_words'], 
        word_embeddings
      )

    return word_embeddings 

  def char_embeddings(self):
    with Path(self.params['chars']).open() as f:
      num_chars = sum(1 for _ in f) + 1
    
    vocab_chars = tf.contrib.lookup.index_table_from_file(
      self.params['chars'], num_oov_buckets=1
    )
    
    char_ids = vocab_chars.lookup(self.chars)
  
    if self.params['pretrained_chars']:
      pretrained_chars = np.load(self.params['char_embeddings'])['embeddings']
      v = np.vstack([pretrained_chars, [[0.] * self.params['dim_chars']]])
      v = tf.Variable(v, dtype=tf.float32, trainable=True)
    else:
      v = tf.get_variable('chars_embeddings', [num_chars + 1, self.params['dim_chars']], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(v, char_ids)
    char_embeddings = self.dropout(char_embeddings)
   
    # Char representations. 
    if self.params['char_representation'] == 'cnn':
      weights = tf.sequence_mask(self.nchars)
      char_embeddings = self.cnn(char_embeddings, weights, self.params['filters'], self.params['kernel_size']) 
    elif self.params['char_representation'] == 'lstm':
      char_embeddings = self.lstm_char_representations(char_embeddings, self.nchars)
    return char_embeddings

  def lstm(self, x):
    lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.params['lstm_size'])
    lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.params['lstm_size'])
    
    (output_fw, output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(
      lstm_cell_fw, lstm_cell_bw, x,
      dtype=tf.float32,
      sequence_length=self.nwords
    )
  
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = self.dropout(output)

    if self.params['pos_embeddings'] == 'lstm':
      output = self.position_embeddings(1600, 2 * self.params['lstm_size'], output)
    return output

  def output_layer(self, x):
    with Path(self.params['tags']).open() as f:
      indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
      num_tags = len(indices) + 1

    logits = tf.layers.dense(x, num_tags)
  
    vocab_tags = tf.contrib.lookup.index_table_from_file(self.params['tags'])
    tags = vocab_tags.lookup(self.labels)
    if self.params['use_crf']:
      crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
      pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, self.nwords)
      log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, tags, self.nwords, crf_params)
      loss = tf.reduce_mean(-log_likelihood, name='loss')
    else:
      pred_ids = tf.argmax(logits, axis=-1)
      labels = tf.one_hot(tags, num_tags)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits
      ), name='loss')

    correct = tf.equal(tf.to_int64(pred_ids), tags)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
    train_step = tf.train.AdamOptimizer(learning_rate=self.params['learning_rate']).minimize(loss)

    reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
      self.params['tags']
    )
    pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))  

  def create(self):
    self.create_placeholders()

    with tf.name_scope('embeddings'):
      if self.params['elmo']:
        word_embeddings = self.elmo_embeddings()
      else:
        word_embeddings = self.word_embeddings()
      char_embeddings = self.char_embeddings()
      embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)
      embeddings = self.dropout(embeddings)
    
    with tf.name_scope('lstm'):
      output = self.lstm(embeddings)

      if self.params['use_attention']:
        output = self.attention(output, output)

    with tf.name_scope('output'):
      self.output_layer(output)
