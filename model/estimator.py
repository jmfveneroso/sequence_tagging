import numpy as np
import re
import json 
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
np.set_printoptions(threshold=np.nan)

params = None

class Estimator:
  def __init__(self):
    self.params = {
      'epochs': 5,
      'current_epoch': 0,
      'best_f1': 0,
    }
    self.learning_rate = 0.001

  def set_dataset_params(self, new_params):
    self.params.update(new_params)

  def load_params_from_file(self, json_file):
    with Path(json_file).open('r') as f:
      data = json.load(f)
      self.params.update(data)
  
    print('===Model parameters===')
    print(json.dumps(self.params, indent=4, sort_keys=True))
    print('\t'.join([str(p) for p in self.params]))
    print('\t'.join([str(self.params[p]) for p in self.params]))

  def restore(self, sess):
    model_base = './checkpoints/model.ckpt'
    saver = tf.train.import_meta_graph(model_base + '.meta')
    sess.run([tf.tables_initializer()])
    saver.restore(sess, model_base)

    with Path(model_base + '.params.json').open('r') as f:
      self.params = json.load(f)
    print('Restoring best model from epoch %s' % (self.params['current_epoch']))

    return saver
  
  def save_model(self, sess, saver):
    model_base = './checkpoints/model.ckpt'
    save_path = saver.save(sess, model_base)

    with Path(model_base + '.params.json').open('w') as f:
      json.dump(self.params, f, indent=4, sort_keys=True)

    print("Model saved in path: %s" % save_path) 

  def get_dict(self, features, labels, train):
    (words, nwords), (chars, nchars), html, (css_chars, css_lengths) = features
    return {
      'inputs/words:0': words,
      'inputs/nwords:0': nwords,
      'inputs/chars:0': chars,
      'inputs/nchars:0': nchars,
      'inputs/html:0': html,
      'inputs/css_chars:0': css_chars,
      'inputs/css_lengths:0': css_lengths,
      'inputs/labels:0': labels,
      'inputs/training:0': train,
      'inputs/learning_rate:0': self.learning_rate,
    }

  def run_epoch(self, sess, filename, train=False, epoch_num=0):
    start_time = time.time()
      
    iterator = DL().input_fn(filename, training=train).make_initializable_iterator()
    next_el = iterator.get_next()
    sess.run(iterator.initializer)
   
    words, tags, preds = [], [], []
    for step in range(1, 1000000):
      try:
        features, labels = sess.run(next_el)
        (words_, nwords_), (chars, nchars), html, (css_chars, css_lengths) = features
  
        target = [
          'output/loss:0', 
          'output/accuracy:0', 
          'output/index_to_string_Lookup:0',
        ]
        if train:
          target.append('output/Adam')

        feed_dict = self.get_dict(features, labels, train)
        r = sess.run(target, feed_dict=feed_dict)
  
        seqlens = nwords_.tolist()
        words += [w[:seqlens[i]] for i, w in enumerate(words_.tolist())]
        tags  += [l[:seqlens[i]] for i, l in enumerate(labels.tolist())]
        preds += [p[:seqlens[i]] for i, p in enumerate(r[2].tolist())]
  
        if step % 50 == 0:
          print('Loss: %.4f, Acc: %.4f, Time: %.4f, Step: %d' % (r[0], r[1], time.time() - start_time, step))
      except tf.errors.OutOfRangeError:
        print('Loss: %.4f, Acc: %.4f, Time: %.4f, Step: %d' % (r[0], r[1], time.time() - start_time, step))
        break
  
    m = evaluate(preds, tags, words)
    print('%s - Epoch %d, Precision: %.4f, Recall: %.4f, F1: %.4f' % (filename, epoch_num, m['precision'], m['recall'], m['f1']))

    if train:
      self.params['current_epoch'] = epoch_num

    return m, (preds, tags, words)

  def train(self, fine_tune=False):
    with tf.Session() as sess:  
      start_time = time.time()

      if fine_tune:
        saver = self.restore(sess)
        self.learning_rate = 0.0001
        self.params['epochs'] = self.params['current_epoch'] + 10
      else:      
        SequenceModel(self.params).create()
        sess.run([tf.initializers.global_variables(), tf.tables_initializer()])
        saver = tf.train.Saver()
      DL().set_params(self.params)
  
      for epoch in range(self.params['current_epoch'], self.params['epochs']):
        _, _ = self.run_epoch(sess, 'train', epoch_num=epoch, train=True)
        _, _ = self.run_epoch(sess, 'test', epoch_num=epoch)
        m, _ = self.run_epoch(sess, 'valid', epoch_num=epoch)
        
        if m['f1'] > self.params['best_f1']:
          self.params['best_f1'] = m['f1']
          self.save_model(sess, saver)

    print('Elapsed time: %.4f' % (time.time() - start_time))
    # if not fine_tune:
    #   self.train(fine_tune=True)
  
  def test(self):
    with tf.Session() as sess:
      _ = self.restore(sess)
      DL().set_params(self.params)
      
      for name in ['train', 'valid', 'test']:
        _, (p, t, w) = self.run_epoch(sess, name)
  
        Path('results/score').mkdir(parents=True, exist_ok=True)
        with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
          for words, preds, tags in zip(w, p, t):
            # f.write(b'-DOCSTART- O O\n\n')
            f.write(b'\n\n')
            for word, pred, tag in zip(words, preds, tags):
              if not word.decode("utf-8") == 'EOS':
                f.write(b' '.join([word, tag, pred]) + b'\n')
              else:
                f.write(b'\n')
  
  def get_alphas(self, filename, doc):
    with tf.Session() as sess:
      _ = self.restore(sess)
  
      features, labels = DL().get_doc(filename, doc)
      (words, nwords), (chars, nchars), html, (css_chars, css_lengths) = features
  
      target = ['output/index_to_string_Lookup:0']
      try:
        tf.get_default_graph().get_tensor_by_name('transformer/alphas:0')
        target.append('transformer/alphas:0')
      except:
        pass
     
      feed_dict = self.get_dict(features, labels, False)
      r = sess.run(target, feed_dict=feed_dict)
 
      words = words[0] 
      preds  = r[0][0]
      labels = labels[0]
      alphas = np.mean(r[1], axis=0) if len(r) > 1 else []
      
      return words, alphas, preds, labels
