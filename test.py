import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pathlib import Path
from model.data_loader import DL
from model.train import train, plot_attention
from model.metrics import evaluate
 
# def get_embeddings(w):
#   words = tf.placeholder(tf.string, shape=(None,), name='words')
#   vocab_words = tf.contrib.lookup.index_table_from_file(
#     str(Path('data/conll2003', 'vocab.words.txt')),
#     num_oov_buckets=1
#   )
#   word_ids = vocab_words.lookup(words)
#   glove = np.load(str(Path('data/conll2003', 'glove.npz')))['embeddings']  # np.array
#   variable = np.vstack([glove, [[0.] * 300]])
#   variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
#   x = tf.nn.embedding_lookup(variable, word_ids)
#   
#   with tf.Session() as sess:
#     sess.run([tf.initializers.global_variables(), tf.tables_initializer()])
#     return sess.run(x, feed_dict={
#       words: w
#     })
# 
# def get_distance(w1, w2):
#   return np.sum(w1 * w2) / (np.sqrt(np.sum(np.square(w1))) * np.sqrt(np.sum(np.square(w2))))
# 
# w = get_embeddings(['ENGLISHMAN', 'Englishman'])
# 
# w[0] = w[0] + w[1] / 2
# d = get_distance(w[0], w[1])
# print(d)

# words, alphas, emb_ids, embs = plot_attention('valid', 0)
# words = [w.decode("utf-8") for w in words]
# 
# print(alphas[:30,:30])

# ===================================
# ===================================

# y = tf.constant([ 
#   [1, 2, 3],
#   [4, 5, 6],
# ], dtype=tf.float64)
# 
# x = tf.constant([2, 2, 2]
# , dtype=tf.float64)
# 
# x = tf.constant([
#   [
#     [ 1, 1, 1, 1 ],
#     [ 2, 2, 2, 2 ],
#     [ 3, 1, 3, 3 ],
#     [ 0, 0, 0, 0 ],
#   ],
#   [
#     [ 0, 0, 0, 0 ],
#     [ 2, 2, 2, 2 ],
#     [ 3, 1, 3, 3 ],
#     [ 0, 0, 0, 0 ],
#   ]
# ], dtype=tf.float64)
# 
# x = tf.expand_dims(x, axis=-1)
# 
# x = tf.tile(x, [1, 2])
# x = tf.matmul(y, x) + x
# 
# y = tf.tile(y, [4])
# y = tf.reshape(y, [4, 4])
# x = tf.concat([x, y], axis=-1)

x = tf.constant([-1, 0, 2, 3], dtype=tf.float64)
x = tf.nn.relu(x)

with tf.Session() as sess:
  sess.run([tf.initializers.global_variables(), tf.tables_initializer()])
  res = sess.run(x)
  print(res)

# ===================================
# ===================================
# 
# 
# with open('results/score/test.preds.txt') as f:
#   preds, labels, words = [], [], []
#   for line in f:
#     l = line.strip().split(' ')
#     if len(l) == 3:
#       labels.append(l[1])
#       preds.append(l[2])
#       words.append(l[0])
# 
#   print(evaluate([preds], [labels], words, verbose=False))
