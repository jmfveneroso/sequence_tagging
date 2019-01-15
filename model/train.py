import numpy as np
import re
import functools
from pathlib import Path
import tensorflow as tf
from model.cnn import masked_conv1d_and_max
import numpy as np
from IPython.display import clear_output, display
import time
from model.data_loader import DL
from model.metrics import evaluate
from model.model import create_model

MAX_TOKEN_LENGTH = 10
DATADIR = 'data/conll2003'
LABEL_COL = 3

# Params
params = {
  'dim_chars': 100,
  'dim': 300,
  'dropout': 0.5,
  'num_oov_buckets': 1,
  'epochs': 25,
  'batch_size': 1,
  'buffer': 15000,
  'filters': 50,
  'kernel_size': 3,
  'lstm_size': 100,
  'words': str(Path(DATADIR, 'vocab.words.txt')),
  'chars': str(Path(DATADIR, 'vocab.chars.txt')),
  'tags': str(Path(DATADIR, 'vocab.tags.txt')),
  'glove': str(Path(DATADIR, 'glove.npz')),
  'fulldoc': True
}

DL().set_params(params)

def process_one_epoch(sess, filename, train=False):  
  start_time = time.time()
    
  iterator = DL().input_fn(filename, training=train).make_initializable_iterator()
  next_el = iterator.get_next()
  sess.run(iterator.initializer)
  
  all_words, all_tags, all_preds, all_seqlens, step = [], [], [], [], 0
  while True:
    try:
      step += 1      
      features, labels_ = sess.run(next_el)
      (words_, nwords_), (chars_, nchars_) = features
      
      target = ['loss/loss:0', 'loss/accuracy:0', 'prediction/index_to_string_Lookup:0']
      # target.append('lstm/alphas:0')
      if train:
        target.append('training/Adam')
    
      
      result = sess.run(target, feed_dict={
        'inputs/words:0': words_,
        'inputs/nwords:0': nwords_,
        'inputs/chars:0': chars_,
        'inputs/nchars:0': nchars_,
        'inputs/labels:0': labels_,
        'inputs/training:0': train
      })
      l = result[0]
      acc = result[1]
      preds_ = result[2]

      # alphas = result[3]
      # print(alphas)
      
      all_words += words_.tolist()        
      all_tags += labels_.tolist()        
      all_preds += preds_.tolist()
      all_seqlens += nwords_.tolist()
      
      cur_time = time.time() - start_time
      msg = 'Loss: %.4f, Accuracy: %.4f, Time: %.4f, Step: %d       ' % (l, acc, cur_time, step)
      print('\r', msg, end='')
    except tf.errors.OutOfRangeError:
      break

  for i in range(len(all_seqlens)):
    all_tags[i] = all_tags[i][:all_seqlens[i]]
    all_preds[i] = all_preds[i][:all_seqlens[i]]    
    all_words[i] = all_words[i][:all_seqlens[i]]    
       
  print()
  return evaluate(all_preds, all_tags, [], verbose=False), (all_words, all_preds, all_tags)

def write_predictions(name):
  with tf.Session() as sess:
    print('Restoring best model...')
    saver = tf.train.import_meta_graph('./checkpoints/model.ckpt.meta')
    sess.run([tf.tables_initializer()])
    saver.restore(sess, "./checkpoints/model.ckpt")

    _, (w, p, t) = process_one_epoch(sess, name)

    Path('results/score').mkdir(parents=True, exist_ok=True)
    with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
      for words, preds, tags in zip(w, p, t):
        for word, pred, tag in zip(words, preds, tags):
          f.write(b' '.join([word, tag, pred]) + b'\n')
        f.write(b'\n')

def train(restore=False):
  if not restore:
    create_model()

  best_f1 = 0
  
  with tf.Session() as sess:  
    if restore:
      saver = tf.train.import_meta_graph('./checkpoints/model.ckpt.meta')
      saver.restore(sess, "./checkpoints/model.ckpt")
    else:
      saver = tf.train.Saver()
    sess.run([tf.initializers.global_variables(), tf.tables_initializer()])

    epochs = 20
    for epoch in range(epochs):
      m, _ = process_one_epoch(sess, 'train', train=True)
      print('TRAIN - Epoch %d, Precision: %.4f, Recall: %.4f, F1: %.4f' % (epoch, m['precision'], m['recall'], m['f1']))

      m, _ = process_one_epoch(sess, 'valid')
      print('VALID - Epoch %d, Precision: %.4f, Recall: %.4f, F1: %.4f' % (epoch, m['precision'], m['recall'], m['f1']))
      
      if m['f1'] > best_f1:
        best_f1 = m['f1']
        save_path = saver.save(sess, "./checkpoints/model.ckpt")
        print("Model saved in path: %s" % save_path)
  
  with tf.Session() as sess:
    print('Restoring best model...')
    saver = tf.train.import_meta_graph('./checkpoints/model.ckpt.meta')
    sess.run([tf.tables_initializer()])
    saver.restore(sess, "./checkpoints/model.ckpt")
    
    m, _ = process_one_epoch(sess, 'valid')
    print('VALID - Precision: %.4f, Recall: %.4f, F1: %.4f' % (m['precision'], m['recall'], m['f1']))
    
    m, _ = process_one_epoch(sess, 'test')
    print('TEST - Precision: %.4f, Recall: %.4f, F1: %.4f' % (m['precision'], m['recall'], m['f1']))

  for name in ['train', 'valid', 'test']:
    write_predictions(name)

if __name__ == '__main__':
  train(restore=False)
