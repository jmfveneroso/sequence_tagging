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
      # Static configurations.
      'datadir': DATADIR,
      'html_tags': str(Path(DATADIR, 'vocab.htmls.txt')),
      'words': str(Path(DATADIR, 'vocab.words.txt')),
      'chars': str(Path(DATADIR, 'vocab.chars.txt')),
      'tags': str(Path(DATADIR, 'vocab.tags.txt')),
      'glove': str(Path(DATADIR, 'glove.npz')),
      'elmo': str(Path(DATADIR, 'elmo.npz')),
      'char_embeddings': str(Path(DATADIR, 'char_embeddings.npz')),
      'dim_chars': 50,
      'dropout': 0.5,
      'filters': 50,
      'kernel_size': 3,
      'char_lstm_size': 25,
      'pretrained_chars': False,
      # General configurations.
      'lstm_size': 200,
      'decoder': 'crf', # crf, logits.
      'char_representation': 'cnn',
      'word_embeddings': 'glove', # glove, elmo. TODO: bert.
      # Attention.
      'use_attention': True,
      'num_blocks': 1,
      'num_heads': 2,
      'pos_embeddings': True,
      'similarity_fn': 'scaled_dot', # rbf, dot, scaled_dot, cosine, bahdanau
      'regularization_fn': 'softmax', # softmax, tanh, linear
      'queries': 'mid_layer', # embeddings, mid_layer
      'keys': 'mid_layer', # embeddings, mid_layer
      'queries_eq_keys': False,
      'mask': True,
      'residual': 'add', # add, concat.
      'layer_normalization': True,
      'html_embeddings': False,
      'css_char_representation': 'cnn'
    }
    params = params if params is not None else {}
    self.params.update(params)

    self.params['words'] = str(Path(self.params['datadir'], 'vocab.words.txt'))
    self.params['chars'] = str(Path(self.params['datadir'], 'vocab.chars.txt'))
    self.params['tags' ] = str(Path(self.params['datadir'], 'vocab.tags.txt'))
    self.params['html_tags'] = str(Path(self.params['datadir'], 'vocab.htmls.txt'))
    self.params['glove'] = str(Path(self.params['datadir'], 'glove.npz'))

    with Path(self.params['tags']).open() as f:
      indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
      self.num_tags = len(indices) + 1

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

  def lstm_char_representations(self, char_embeddings, nchars, scope='lstm_chars'):
    with tf.variable_scope(scope):
      dim_words = tf.shape(char_embeddings)[1]
      dim_chars = tf.shape(char_embeddings)[2]

      t = tf.reshape(char_embeddings, [-1, dim_chars, self.params['dim_chars']])
  
      lstm_cell_fw_c = tf.nn.rnn_cell.LSTMCell(self.params['char_lstm_size'])
      lstm_cell_bw_c = tf.nn.rnn_cell.LSTMCell(self.params['char_lstm_size'])
      
      (_, _), (output_fw, output_bw) = tf.nn.bidirectional_dynamic_rnn(
        lstm_cell_fw_c, lstm_cell_bw_c, t,
        dtype=tf.float32,
        sequence_length=tf.reshape(nchars, [-1]),
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
  
  def attention(self, queries, keys):
    attention_size = queries.shape[2].value
    output_size = keys.shape[2].value
  
    Q = tf.layers.dense(queries, attention_size)
    if self.params['queries_eq_keys']:
      K = Q
    else:
      K = tf.layers.dense(keys, attention_size)
    V = tf.layers.dense(keys, output_size)

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
    if self.params['mask']:
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
    if self.params['mask']:
      query_masks = tf.sign(tf.reduce_sum(tf.abs(Q), axis=-1))
      query_masks = tf.tile(query_masks, [num_heads, 1])
      query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(K)[1]])
      alphas *= query_masks
  
    alphas = self.dropout(alphas)
    outputs = tf.matmul(alphas, V_)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

    if self.params['residual'] == 'add':
      outputs += keys
    elif self.params['residual'] == 'concat':
      outputs = tf.concat([outputs, keys], axis=-1)

    # Layer normalization.
    if self.params['layer_normalization']:
      outputs = self.normalize(outputs)
    return outputs
  
  def pos_embeddings(self, inputs, emb_dim, max_length=1600):
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
      self.words    = tf.placeholder(tf.string,  shape=(None, None),       name='words'   )
      self.nwords   = tf.placeholder(tf.int32,   shape=(None,),            name='nwords'  )
      self.chars    = tf.placeholder(tf.string,  shape=(None, None, None), name='chars'   )
      self.nchars   = tf.placeholder(tf.int32,   shape=(None, None),       name='nchars'  )
      self.html     = tf.placeholder(tf.string,  shape=(None, None, None), name='html'    )
      self.css_chars   = tf.placeholder(tf.string,  shape=(None, None, None), name='css_chars'   )
      self.css_lengths = tf.placeholder(tf.int32,   shape=(None, None),       name='css_lengths'  )
      self.labels   = tf.placeholder(tf.string,  shape=(None, None),       name='labels'  )
      self.training = tf.placeholder(tf.bool,    shape=(),                 name='training')
      self.learning_rate = tf.placeholder_with_default(
        0.001, shape=(), name='learning_rate'      
      )
 
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

    return word_embeddings, 1024

  def html_embeddings(self):
    with Path(self.params['html_tags']).open() as f:
      num_html_tags = sum(1 for _ in f) + 1
    
    vocab_chars = tf.contrib.lookup.index_table_from_file(
      self.params['chars'], num_oov_buckets=1
    )
 
    html_tags = tf.slice(self.html, [0, 0, 0], [-1, -1, 2])
    html_tag_ids = vocab_chars.lookup(html_tags)

    html_embedding_size = 32
    v = tf.get_variable('html_embeddings', [num_html_tags + 1, html_embedding_size], tf.float32)
    html_embeddings = tf.nn.embedding_lookup(v, html_tag_ids)
    timesteps = tf.shape(html_tags)[1]
    html_embeddings = tf.reshape(html_embeddings, [-1, timesteps, html_embedding_size*2])
    return html_embeddings, html_embedding_size * 2

  def css_embeddings(self):
    with Path(self.params['chars']).open() as f:
      num_chars = sum(1 for _ in f) + 1
    
    vocab_chars = tf.contrib.lookup.index_table_from_file(
      self.params['chars'], num_oov_buckets=1
    )
    
    char_ids = vocab_chars.lookup(self.css_chars)
  
    v = tf.get_variable('chars_embeddings2', [num_chars + 1, self.params['dim_chars']], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(v, char_ids)
    char_embeddings = self.dropout(char_embeddings)
  
    emb_size = 0
    if self.params['css_char_representation'] == 'cnn':
      weights = tf.sequence_mask(self.css_lengths)
      char_embeddings = self.cnn(char_embeddings, weights, self.params['filters'], self.params['kernel_size']) 
      emb_size = self.params['filters']
    elif self.params['css_char_representation'] == 'lstm':
      char_embeddings = self.lstm_char_representations(char_embeddings, self.css_lengths, scope='css_lstm')
      emb_size = self.params['char_lstm_size'] * 2
    return char_embeddings, emb_size

  def glove_embeddings(self):
    vocab_words = tf.contrib.lookup.index_table_from_file(
      self.params['words'], num_oov_buckets=1
    )

    word_ids = vocab_words.lookup(self.words)
    glove = np.load(self.params['glove'])['embeddings']
    variable = np.vstack([glove, [[0.] * 300]])
    variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

    return word_embeddings, 300

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
      v = tf.Variable(v, dtype=tf.float32, trainable=False)
    else:
      v = tf.get_variable('chars_embeddings', [num_chars + 1, self.params['dim_chars']], tf.float32)
    char_embeddings = tf.nn.embedding_lookup(v, char_ids)
    char_embeddings = self.dropout(char_embeddings)
  
    emb_size = 0
    if self.params['char_representation'] == 'cnn':
      weights = tf.sequence_mask(self.nchars)
      char_embeddings = self.cnn(char_embeddings, weights, self.params['filters'], self.params['kernel_size']) 
      emb_size = self.params['filters']
    elif self.params['char_representation'] == 'lstm':
      char_embeddings = self.lstm_char_representations(char_embeddings, self.nchars)
      emb_size = self.params['char_lstm_size'] * 2
    return char_embeddings, emb_size

  def embeddings_layer(self):
    word_emb_size = 0
    if self.params['word_embeddings'] == 'elmo':
      word_embs, word_emb_size = self.elmo_embeddings()
    elif self.params['word_embeddings'] == 'glove':
      word_embs, word_emb_size = self.glove_embeddings()
    else:
      raise Exception('No word embeddings were selected.')

    embs = [word_embs]
    emb_size = word_emb_size
    if not self.params['char_representation'] == 'none':
      char_embs, char_emb_size = self.char_embeddings()
      embs.append(char_embs)
      emb_size += char_emb_size

    if self.params['html_embeddings']:
      html_embs, html_embs_size = self.html_embeddings()
      css_embs, css_embs_size = self.css_embeddings()
      embs.append(html_embs)
      embs.append(css_embs)
      emb_size += html_embs_size + css_embs_size

    embs = tf.concat(embs, axis=-1)
    embs = self.dropout(embs)
    return embs, emb_size

  def lstm(self, x, lstm_size, var_scope='lstm'):
    with tf.variable_scope(var_scope):
      lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(lstm_size)
      lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(lstm_size)
      
      (output_fw, output_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(
        lstm_cell_fw, lstm_cell_bw, x,
        dtype=tf.float32,
        sequence_length=self.nwords
      )
  
      output = tf.concat([output_fw, output_bw], axis=-1)
      output = self.dropout(output)
      return output

  def output_layer(self, x):
    logits = tf.layers.dense(x, self.num_tags)

    vocab_tags = tf.contrib.lookup.index_table_from_file(self.params['tags'])
    tags = vocab_tags.lookup(self.labels)
    if self.params['decoder'] == 'crf':
      crf_params = tf.get_variable("crf", [self.num_tags, self.num_tags], dtype=tf.float32)
      pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, self.nwords)
      log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, tags, self.nwords, crf_params)
      loss = tf.reduce_mean(-log_likelihood, name='loss')
    else:
      pred_ids = tf.argmax(logits, axis=-1)
      labels = tf.one_hot(tags, self.num_tags)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits
      ), name='loss')

    correct = tf.equal(tf.to_int64(pred_ids), tags)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
    train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

    reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
      self.params['tags']
    )
    pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))  

  def mid_layer(self, x):
    return self.lstm(x, self.params['lstm_size']), 2 * self.params['lstm_size']

  def single_block_transformer(self, x, x_size):
    if self.params['queries'] == self.params['keys'] == 'embeddings':
      if self.params['pos_embeddings']:
        x = self.pos_embeddings(x, x_size)
      x = self.attention(x, x)
      output, _ = self.mid_layer(x)
      return output

    else:
      mid_output, mid_size = self.mid_layer(x)

      pos_emb_size = 0
      if self.params['queries'] == 'embeddings':
        queries = x 
        pos_emb_size = x_size
      elif self.params['queries'] == 'mid_layer':
        queries = mid_output
        pos_emb_size = mid_size
      else:
        raise Exception('Invalid queries value')

      if self.params['pos_embeddings']:
        queries = self.pos_embeddings(queries, pos_emb_size)

      return self.attention(queries, mid_output)

  def transformer(self, x, x_size):
    for i in range(self.params['num_blocks']):
      if self.params['pos_embeddings']:
        outputs = self.pos_embeddings(x, x_size)
      outputs = self.attention(outputs, outputs)

      # Bidirectional LSTM will output a tensor with a shape that is twice 
      # the hidden layer size.
      x = self.lstm(x, x_size / 2, var_scope='transformer_' + str(i)) + x 
    return x 

  def create(self):
    self.create_placeholders()

    with tf.name_scope('embeddings'):
      embs, emb_size = self.embeddings_layer()
   
    if self.params['use_attention']:
      with tf.name_scope('transformer'):
        if self.params['num_blocks'] == 1:
            output = self.single_block_transformer(embs, emb_size) 
        else:
          output = self.transformer(embs, emb_size) 
    else:
      output, _ = self.mid_layer(embs)

    with tf.name_scope('output'):
      self.output_layer(output)
