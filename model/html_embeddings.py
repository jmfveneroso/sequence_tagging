import tensorflow as tf
from pathlib import Path
from model.char_representations import get_char_embeddings, lstm_char_representations

def get_html_embeddings(html, html_vocab_file):
  with Path(html_vocab_file).open() as f:
    num_html_tags = sum(1 for _ in f) + 1
  
  vocab_html = tf.contrib.lookup.index_table_from_file(
    html_vocab_file, num_oov_buckets=1
  )

  html_tags = tf.slice(html, [0, 0, 0], [-1, -1, 2])
  html_tag_ids = vocab_html.lookup(html_tags)

  html_embedding_size = 50
  v = tf.get_variable('html_embeddings', [num_html_tags + 1, html_embedding_size], tf.float32)
  html_embeddings = tf.nn.embedding_lookup(v, html_tag_ids)
  timesteps = tf.shape(html_tags)[1]
  html_embeddings = tf.reshape(html_embeddings, [-1, timesteps, html_embedding_size*2])
  return html_embeddings

def get_css_embeddings(css_chars, css_lengths, char_vocab_file, training=False):
  lstm_size = 25
  char_embedding_size = 50

  char_embeddings = get_char_embeddings(css_chars, char_vocab_file, char_embedding_size, training=training)
  char_embeddings = tf.reduce_mean(char_embeddings, axis=-2)
  return char_embeddings

def get_html_representations(html, html_vocab_file, css_chars, css_lengths, char_vocab_file, training=False):
  html_embs = get_html_embeddings(html, html_vocab_file)
  css_embs = get_css_embeddings(
    css_chars, css_lengths,
    char_vocab_file, training=training
  )

  html_embs = tf.concat([html_embs, css_embs], axis=-1)
  return html_embs

