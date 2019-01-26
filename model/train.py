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
import subprocess

MAX_TOKEN_LENGTH = 10
DATADIR = 'data/conll2003'
LABEL_COL = 3

# Params
params = {
  'epochs': 50,
  'batch_size': 1,
  'buffer': 15000,
  'fulldoc': True
}

import numpy
numpy.set_printoptions(threshold=numpy.nan)

DL().set_params(params)

def process_one_epoch(sess, filename, train=False, show_alphas=False):  
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
      if show_alphas:
        target.append('lstm/alphas:0')
        target.append('lstm/lstm_states:0')
        target.append('lstm/weighted_lstm_states:0')
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

      if show_alphas:
        alphas = result[3]
        states = result[4]
        w_states = result[5]
        print(states.tolist())
        print(w_states.tolist())
        print(alphas.tolist())
      
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
        f.write(b'-DOCSTART- O O\n\n')
        for word, pred, tag in zip(words, preds, tags):
          if not word.decode("utf-8") == 'EOS':
            f.write(b' '.join([word, tag, pred]) + b'\n')
          else:
            f.write(b'\n')

def test():
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
  result = subprocess.check_output(['sh', './eval.sh'])
  # print(result)

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

    epochs = params['epochs']
    for epoch in range(epochs):
      m, _ = process_one_epoch(sess, 'train', train=True)
      print('TRAIN - Epoch %d, Precision: %.4f, Recall: %.4f, F1: %.4f' % (epoch, m['precision'], m['recall'], m['f1']))

      m, _ = process_one_epoch(sess, 'valid')
      print('VALID - Epoch %d, Precision: %.4f, Recall: %.4f, F1: %.4f' % (epoch, m['precision'], m['recall'], m['f1']))
      
      if m['f1'] > best_f1:
        best_f1 = m['f1']
        save_path = saver.save(sess, "./checkpoints/model.ckpt")
        print("Model saved in path: %s" % save_path) 
  test()

def quick_train(filename, docs):
  if not isinstance(docs, list):
    docs = [docs] 

  create_model()

  with tf.Session() as sess:  
    saver = tf.train.Saver()
    sess.run([tf.initializers.global_variables(), tf.tables_initializer()])

    f1 = 0
    epochs = 150
    for epoch in range(epochs):
      start_time = time.time()
      for doc in docs:
        features, labels_ = [(f, l) for (f, l) in DL().generator_fn(filename)][doc]
        (words_, nwords_), (chars_, nchars_) = features
    
        target = [
          'loss/loss:0', 'loss/accuracy:0', 
          'prediction/index_to_string_Lookup:0', 
          'training/Adam'
        ]
        
        result = sess.run(target, feed_dict={
          'inputs/words:0':  [words_],
          'inputs/nwords:0': [nwords_],
          'inputs/chars:0':  [chars_],
          'inputs/nchars:0': [nchars_],
          'inputs/labels:0': [labels_],
          'inputs/training:0': False
        })
        l = result[0]
        acc = result[1]
        preds_ = result[2]

        cur_time = time.time() - start_time

        all_words = words_
        all_tags = labels_
        all_preds = preds_.tolist()
        all_seqlens = nwords_

        all_tags  = [all_tags[:nwords_]]
        all_preds[0] = all_preds[0][:nwords_]
        all_words = all_words[:nwords_]   

        m = evaluate(all_preds, all_tags, [], verbose=False)
        f1 = m['f1']
        msg = 'Loss: %.4f, Accuracy: %.4f, F1: %.4f, Time: %.4f, Step: %d       ' % (l, acc, f1, cur_time, epoch)
        print('\r', msg, end='')

        if f1 > 0.9:
          break
      if f1 > 0.9:
        break
    save_path = saver.save(sess, "./checkpoints/model2.ckpt")
    print("Model saved in path: %s" % save_path)

def plot_attention(filename, doc, checkpoint=''):
  with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./checkpoints/model' + checkpoint + '.ckpt.meta')
    sess.run([tf.tables_initializer()])
    saver.restore(sess, './checkpoints/model' + checkpoint + '.ckpt')

    features, labels_ = [(f, l) for (f, l) in DL().generator_fn(filename)][doc]
    (words_, nwords_), (chars_, nchars_) = features
    
    target = [
      'lstm/alphas0:0', 
      'embeddings/string_to_index_Lookup:0',
      'embeddings/embedding_lookup_1/Identity:0',
      'prediction/index_to_string_Lookup:0',
      'loss/loss:0', 'loss/accuracy:0', 
      'lstm/alphas1:0', 
      # 'lstm/alphas2:0', 
      # 'lstm/alphas3:0', 
    ]
     
    result = sess.run(target, feed_dict={
      'inputs/words:0':  [words_],
      'inputs/nwords:0': [nwords_],
      'inputs/chars:0':  [chars_],
      'inputs/nchars:0': [nchars_],
      'inputs/labels:0': [labels_],
      'inputs/training:0': False
    })
    alphas = result[0][0]
    word_emb_ids = result[1]
    word_embs = result[2]
    preds_ = result[3]
    loss = result[4]
    acc = result[5]
    alphas1 = result[6][0]
    # alphas2 = result[7][0]
    # alphas3 = result[8][0]
    print('Loss: %f, Accuracy: %f' % (loss, acc))

    # alphas = (alphas + alphas1 + alphas2 + alphas3) / 4.0
    alphas = (alphas + alphas1) / 2.0

    return words_, alphas, word_emb_ids, word_embs, preds_[0], labels_
