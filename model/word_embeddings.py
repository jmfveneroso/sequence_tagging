import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def glove(words, words_vocab_file, glove_file):
  vocab_words = tf.contrib.lookup.index_table_from_file(
    words_vocab_file, num_oov_buckets=1
  )

  word_ids = vocab_words.lookup(words)
  glove = np.load(glove_file)['embeddings']
  variable = np.vstack([glove, [[0.] * 300]])
  variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
  word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

  return word_embeddings

def word2vec(words, words_vocab_file, w2v_file):
  vocab_words = tf.contrib.lookup.index_table_from_file(
    words_vocab_file, num_oov_buckets=1
  )

  word_ids = vocab_words.lookup(words)
  w2v = np.load(w2v_file)['embeddings']
  variable = np.vstack([w2v, [[0.] * 300]])
  variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
  word_embeddings = tf.nn.embedding_lookup(variable, word_ids)

  return word_embeddings

def elmo(words, nwords):
  elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
  
  word_embeddings = elmo(
    inputs={
      "tokens": words,
      "sequence_len": nwords
    },
    signature="tokens",
    as_dict=True
  )["elmo"]

  return word_embeddings
