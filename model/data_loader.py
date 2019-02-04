import re
import functools
import json
import math
import tensorflow as tf
from pathlib import Path

def get_sentences(f): 
  filename = f.parts[-1] 
  with f.open('r', encoding="utf-8") as f:
    uid_start = {
      'train': 1,
      'valid': 204568,
      'test': 256146,
    }
    # 302812
    uid = uid_start[filename] 

    sentences = f.read().strip().split('\n\n')
    sentences = [[t.split() for t in s.split('\n')] for s in sentences if len(s) > 0] 

    for i, s in enumerate(sentences):
      for j, t in enumerate(sentences[i]):
        sentences[i][j].append(uid)
        uid += 1
    return sentences

def split_array(arr, separator_fn):
  arrays = [[]]
  for i, el in enumerate(arr):
    if separator_fn(el):
      arrays.append([])
    else:
      arrays[-1].append(el)
  return [a for a in arrays if len(a) > 0]

def join_arrays(arr, separator):
  res = []
  for i, el in enumerate(arr):
    res += el + [separator]
  return res

def group_by_n(doc, separator_fn, n):
  arr = []
  for i in range(0, len(doc), n):
    eos = ['EOS', '-X-', '-X-', 'O']
    arr.append(join_arrays(doc[i:i+n], eos))
  return arr

def pad_array(arr, padding, max_len):
  return arr + [padding] * (max_len - len(arr))

class DL:
  __instance = None
  def __new__(cls, *args, **kwargs):
    if not DL.__instance:
      DL.__instance = DL.__DataLoader(*args, **kwargs)
    return DL.__instance 

  def __getattr__(self, name):
    return getattr(self.instance, name)

  class __DataLoader:
    def __init__(self):
      self.params = {
        'label_col': 3,
        'batch_size': 1,
        'buffer': 15000,
        'datadir': 'data/conll2003',
        'fulldoc': False,
        'splitsentence': False,
      }

    def __str__(self):
      return repr(self)

    def save_params(self):
      with Path(self.params['datadir'], 'params.json').open('w') as f:
        json.dump(self.params, f, indent=2, sort_keys=True)

    def parse_sentence(self, sentence):
      # Encode in Bytes for Tensorflow.
      words = [s[0] for s in sentence]
      uids = [s[4] for s in sentence]

      tags = [s[self.params['label_col']].encode() for s in sentence]
      
      # Chars.
      chars = [[c.encode() for c in w] for w in words]
      lengths = [len(c) for c in chars]
      chars = [pad_array(c, b'<pad>', max(lengths)) for c in chars]
      
      words = [s[0].encode() for s in sentence]    
      return ((words, uids, len(words)), (chars, lengths)), tags
      
    def generator_fn(self, filename):
      sentences = get_sentences(Path(self.params['datadir'], filename))

      if self.params['fulldoc']:
        separator_fn = lambda el : el[0][0] == '-DOCSTART-'
        eos = ['EOS', '-X-', '-X-', 'O', 0]

        documents = split_array(sentences, separator_fn)
 
        if self.params['splitsentence']:
          documents = [group_by_n(d, lambda el : el[0] == 'EOS', 5) for d in documents]
          documents = [s for d in documents for s in d]
        else:
          documents = [join_arrays(d, eos) for d in documents]

        for i, d in enumerate(documents):
          yield self.parse_sentence(d)
      else:
        for s in sentences:
          yield self.parse_sentence(s)
          
    def input_fn(self, filename, training=False):
      shapes = (
       (([None], [None], ()),   # (words, uids, nwords)
       ([None, None], [None])), # (chars, nchars)  
       [None]                   # tags
      )
    
      types = (
        ((tf.string, tf.int32, tf.int32),
        (tf.string, tf.int32)),  
        tf.string
      )
    
      defaults = (
        (('<pad>', 0, 0),
        ('<pad>', 0)), 
        'O'
      )
    
      dataset = tf.data.Dataset.from_generator(
        functools.partial(self.generator_fn, filename),
        output_types=types, output_shapes=shapes
      )
    
      # if training:
      #   dataset = dataset.shuffle(self.params['buffer'])
   
      batch_size = self.params.get('batch_size', 20)
      return dataset.padded_batch(batch_size, shapes, defaults)

    def get_doc(self, filename, doc):
      old_param = self.params['fulldoc']
      old_param2 = self.params['splitsentence']
      self.params['fulldoc'] = True
      self.params['splitsentence'] = False
      features, labels = [(f, l) for (f, l) in DL().generator_fn(filename)][doc]
      self.params['fulldoc'] = old_param
      self.params['splitsentence'] = old_param2
      (words, nwords), (chars, nchars) = features
      features = ([words], [nwords]), ([chars], [nchars])
      return features, [labels]

    def set_params(self, params=None):
      params = params if params is not None else {}
      self.params.update(params)

      if self.params['fulldoc']:
        self.params['batch_size'] = 1
