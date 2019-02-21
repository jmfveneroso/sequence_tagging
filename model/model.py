import math
import numpy as np 
import tensorflow as tf
from pathlib import Path
import tensorflow_hub as hub
from model.char_representations import get_char_representations, get_char_embeddings
from model.attention import attention
from model.word_embeddings import glove, elmo
from model.html_embeddings import get_html_representations

class SequenceModel:
  def __init__(self, params=None):
    self.params = {
      # Static configurations.
      'datadir': 'data/conll2003',
      # General configurations.
      'lstm_size': 200,
      'decoder': 'crf', # crf, logits.
      'char_representation': 'cnn',
      'word_embeddings': 'glove', # glove, elmo. TODO: bert.
      'model': 'lstm_crf', # lstm_crf, html_attention, self_attention, transformer
    }
    params = params if params is not None else {}
    self.params.update(params)

    self.params['words']     = str(Path(self.params['datadir'], 'vocab.words.txt'))
    self.params['chars']     = str(Path(self.params['datadir'], 'vocab.chars.txt'))
    self.params['tags' ]     = str(Path(self.params['datadir'], 'vocab.tags.txt'))
    self.params['html_tags'] = str(Path(self.params['datadir'], 'vocab.htmls.txt'))
    self.params['glove']     = str(Path(self.params['datadir'], 'glove.npz'))

    with Path(self.params['tags']).open() as f:
      indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
      self.num_tags = len(indices) + 1
  
  def dropout(self, x):
    return tf.layers.dropout(x, rate=0.5, training=self.training)

  def create_placeholders(self):
    with tf.name_scope('inputs'):
      self.words         = tf.placeholder(tf.string,  shape=(None, None),       name='words'       )
      self.nwords        = tf.placeholder(tf.int32,   shape=(None,),            name='nwords'      )
      self.chars         = tf.placeholder(tf.string,  shape=(None, None, None), name='chars'       )
      self.nchars        = tf.placeholder(tf.int32,   shape=(None, None),       name='nchars'      )
      self.html          = tf.placeholder(tf.string,  shape=(None, None, None), name='html'        )
      self.css_chars     = tf.placeholder(tf.string,  shape=(None, None, None), name='css_chars'   )
      self.css_lengths   = tf.placeholder(tf.int32,   shape=(None, None),       name='css_lengths' )
      self.labels        = tf.placeholder(tf.string,  shape=(None, None),       name='labels'      )
      self.training      = tf.placeholder(tf.bool,    shape=(),                 name='training'    )
      self.learning_rate = tf.placeholder_with_default(
        0.001, shape=(), name='learning_rate'      
      )
 
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
    with tf.name_scope('output'):
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
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
          labels=labels, logits=logits
        ), name='loss')

      correct = tf.equal(tf.to_int64(pred_ids), tags)
      accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
      train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

      reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
        self.params['tags']
      )
      pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))  

  def lstm_crf(self, word_embs='glove', char_embs='cnn'):
    if word_embs == 'elmo':
      word_embs = elmo(self.words, self.nwords)
    elif word_embs == 'glove':
      word_embs = glove(self.words, self.params['words'], self.params['glove'])
    else:
      raise Exception('No word embeddings were selected.')

    embs = [word_embs]
    if char_embs in ['cnn', 'lstm']:
      char_embs = get_char_representations(
        self.chars, self.nchars, 
        self.params['chars'], mode=char_embs,
        training=self.training
      )
      embs.append(char_embs)
    embs = tf.concat(embs, axis=-1)
    embs = self.dropout(embs)
    return self.lstm(embs, self.params['lstm_size'])

  def self_attention(self, num_heads=5, residual='add', queries_eq_keys=False):
    word_embs = glove(self.words, self.params['words'], self.params['glove'])
    char_embs = get_char_representations(
      self.chars, self.nchars, 
      self.params['chars'], mode='lstm',
      training=self.training
    )
    html_embs = get_html_representations(
      self.html, self.params['html_tags'],
      self.css_chars, self.css_lengths,
      self.params['chars'], training=self.training
    )

    embs = tf.concat([word_embs, char_embs, html_embs], axis=-1)
    embs = self.dropout(embs)
    output = self.lstm(embs, self.params['lstm_size'])

    return attention(
      output, output, num_heads,
      residual=residual, queries_eq_keys=queries_eq_keys,
      training=self.training
    )

  def html_attention(self, num_heads=5, residual='concat'):
    word_embs = glove(self.words, self.params['words'], self.params['glove'])
    char_embs = get_char_representations(
      self.chars, self.nchars, 
      self.params['chars'], mode='lstm',
      training=self.training
    )

    embs = tf.concat([word_embs, char_embs], axis=-1)
    embs = self.dropout(embs)
    output = self.lstm(embs, self.params['lstm_size'])

    html_embs = get_html_representations(
      self.html, self.params['html_tags'],
      self.css_chars, self.css_lengths,
      self.params['chars'], training=self.training
    )
    html_embs = self.dropout(html_embs)

    return attention(
      html_embs, output, num_heads,
      residual=residual, queries_eq_keys=False,
      training=self.training
    )

  def transformer(self, num_blocks=2, num_heads=5, mid_layer='lstm'):
    word_embs = glove(self.words, self.params['words'], self.params['glove'])
    char_embs = get_char_representations(
      self.chars, self.nchars, 
      self.params['chars'], mode='lstm',
      training=self.training
    )

    embs = tf.concat([word_embs, char_embs], axis=-1)
    x = self.dropout(embs)

    for i in range(num_blocks):
      x = attention(
        x, x, num_heads, residual='add', queries_eq_keys=False,
        training=self.training
      )

      # Bidirectional LSTM will output a tensor with a shape that is twice 
      # the hidden layer size.
      if mid_layer == 'feed_forward':
        x = tf.layers.dense(tf.layers.dense(x, 200), 200)
      elif mid_layer == 'lstm':
        x = self.lstm(x, x.shape[2].value / 2, var_scope='transformer_' + str(i)) + x 
    return x 

  def create(self):
    self.create_placeholders()

    model = self.params['model']
    if model == 'lstm_crf':
      output = self.lstm_crf(char_embs=self.params['char_representation'])
    elif model == 'html_attention':
      output = self.html_attention()
    elif model == 'self_attention':
      output = self.self_attention()
    elif model == 'transformer':
      output = self.transformer()
    else:
      raise Exception('Model does not exist.')

    self.output_layer(output)
