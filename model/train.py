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
from model.model import SequenceModel
import subprocess
np.set_printoptions(threshold=np.nan)

fulldoc = True

DL().set_params({
  'epochs': 10,
  'fulldoc': fulldoc
})

def create_model():
  params = {
    'lstm_size': 200,
    'char_representation': 'cnn',
    'use_attention': True,
    'use_crf': True,
    'num_heads': 2,
    'similarity_fn': 'scaled_dot',
    'regularization_fn': 'softmax',
    'pos_embeddings': 'lstm',
  }
  SequenceModel(params).create()
  print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
    fulldoc, params['lstm_size'], params['char_representation'],
    'crf' if params['use_crf'] else 'argmax', params['use_attention'],
    params['similarity_fn'], params['regularization_fn'],
    params['pos_embeddings'], params['num_heads']
  ))

def run_step(sess, features, labels, train=False, alphas=False):
  (words, nwords), (chars, nchars) = features

  target = [
    'output/loss:0', 
    'output/accuracy:0', 
    'output/index_to_string_Lookup:0'
  ]

  num_heads = 0
  if alphas:
    try: 
      for i in range(0, 8):
        tf.get_default_graph().get_tensor_by_name('lstm/alphas' + str(i) + ':0')
        target.append('lstm/alphas' + str(i) + ':0')
        num_heads += 1
    except:
      pass

  if train: 
    target.append('output/Adam')
  
  result = sess.run(target, feed_dict={
    'inputs/words:0': words,
    'inputs/nwords:0': nwords,
    'inputs/chars:0': chars,
    'inputs/nchars:0': nchars,
    'inputs/labels:0': labels,
    'inputs/training:0': train
  })
  return result[:3], result[3:3+num_heads]

def print_stats(loss, accuracy, time, step, f1=None):
  if f1 is None:
    msg = 'Loss: %.4f, Accuracy: %.4f, Time: %.4f, Step: %d       ' % (loss, accuracy, time, step)
  else:
    msg = 'Loss: %.4f, Accuracy: %.4f, F1: %.4f, Time: %.4f, Step: %d       ' % (loss, accuracy, f1, time, step)
  print('\r', msg, end='')

def restore(sess, checkpoint=''):
  saver = tf.train.import_meta_graph('./checkpoints/model' + checkpoint + '.ckpt.meta')
  sess.run([tf.tables_initializer()])
  saver.restore(sess, './checkpoints/model' + checkpoint + '.ckpt')
  return saver

def process_one_epoch(sess, filename, train=False):  
  start_time = time.time()
    
  iterator = DL().input_fn(filename, training=train).make_initializable_iterator()
  next_el = iterator.get_next()
  sess.run(iterator.initializer)
  
  all_words, all_tags, all_preds, all_seqlens, step = [], [], [], [], 1
  while True:
    try:
      features, labels = sess.run(next_el)
      (words, nwords), (_, _) = features

      r, _ = run_step(sess, features, labels, train)

      all_seqlens += nwords.tolist()
      all_words += words.tolist()        
      all_tags += labels.tolist()        
      all_preds += r[2].tolist()
      
      print_stats(r[0], r[1], time.time() - start_time, step)
      step += 1      
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

def train(restore=False):
  start_time = time.time()

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
      f1 = m['f1']

      m, _ = process_one_epoch(sess, 'test')
      print('TEST - Epoch %d, Precision: %.4f, Recall: %.4f, F1: %.4f' % (epoch, m['precision'], m['recall'], m['f1']))
      
      if f1 > best_f1:
        best_f1 = f1
        save_path = saver.save(sess, "./checkpoints/model.ckpt")
        print("Model saved in path: %s" % save_path) 
  test()
  print('Elapsed time: %.4f' % (time.time() - start_time))

def quick_train(filename, doc, epochs=150):
  create_model()

  with tf.Session() as sess:  
    saver = tf.train.Saver()
    sess.run([tf.initializers.global_variables(), tf.tables_initializer()])

    for step in range(epochs):
      start_time = time.time()
      features, labels = DL().get_doc(filename, doc)
      r, _ = run_step(sess, features, labels, train=True)

      tags = labels
      preds = r[2].tolist()

      m = evaluate(preds, tags, [], verbose=False)
      print_stats(r[0], r[1], time.time() - start_time, step, f1=m['f1'])

      if m['f1'] > 0.95:
        break
    save_path = saver.save(sess, "./checkpoints/model2.ckpt")
    print("Model saved in path: %s" % save_path)

def get_alphas(filename, doc, checkpoint=''):
  with tf.Session() as sess:
    saver = restore(sess, checkpoint=checkpoint)

    features, labels = DL().get_doc(filename, doc)
    (words, _), (_, _) = features

    r, alphas = run_step(sess, features, labels, alphas=True)
    preds  = r[2][0]
 
    num_heads = len(alphas)
    if num_heads > 0:
      alphas = [a[0] for a in alphas]
      alphas = np.sum(alphas, axis=0) / num_heads

    return words[0], alphas, preds, labels[0]
