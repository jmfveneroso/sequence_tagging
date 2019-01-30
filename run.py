import matplotlib
matplotlib.use('Agg')
from model.train import train, get_alphas, quick_train, test
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable debug logs Tensorflow.
tf.logging.set_verbosity(tf.logging.ERROR)

gcfg = [
  ('ASHES'     , 'valid', 2  , 3  , 14 , 719  , 730  ),
  ('KENT'      , 'valid', 113, 4  , 18 , 31781, 31795),
  ('Ireland'   , 'test' , 230, 22 , 52 , 50248, 50275),
  ('ENGLISHMAN', 'test' , 230, 2  , 29 , 50071, 50098),
  ('Doetinchem', 'test' , 228, 149, 153, 49770, 49774),
  ('VOLENDAM'  , 'test' , 228, 4  , 36 , 49625, 49657),
  ('Barcelona' , 'test' , 227, 36 , 61 , 49565, 49597),
  ('RIBALTA'   , 'test' , 225, 4  , 21 , 49392, 49409),
]

def matrix(checkpoint, example, print_all=False, verbose=True):
  global gcfg
  cfg = gcfg[example]

  words, alphas_, preds, labs = get_alphas(cfg[1], cfg[2], checkpoint=checkpoint)
  words = [w.decode("utf-8") for w in words]

  first = cfg[3] 
  second = cfg[4]
  p1 = preds[first].decode("utf-8")
  p2 = preds[second].decode("utf-8")
  label = labs[first].decode("utf-8")
  tick = 'Y' if p1 == p2 == label else 'N'
  print('[%s] %s (%s) %s (%s) should be %s' % (tick, words[first], p1, words[second], p2, label))

  if len(alphas_) > 0:
    padding = 5
    num_words = 30
    alphas = alphas_[cfg[3]]
    start = cfg[3] - padding
    end = cfg[4] + padding
    if start < 0: start = 0
    if end > len(words): end = len(words)
    
    if print_all:
      start = 0
      end = len(words)

    if verbose:
      for i, (w, p, l, a) in enumerate(zip(words, labs, preds, alphas)):
        if i >= start and i <= end:
          prefix = '>>>' if i == cfg[3] or i == cfg[4] else ''
          print(prefix, i, w, p, l, a)
    
    fig = plt.figure(figsize=(10.0, 10.0), dpi=600)
    plt.title()
    plt.xticks(range(0,end), words[start:end], rotation='vertical') 
    plt.yticks(range(0,end), words[start:end])
        
    plt.imshow(alphas_[start:end,start:end])
    plt.savefig('figures/' + str(example) + '.png', dpi=600)

def test_pairs():
  valid, test = [], []
  with open('results/score/valid.preds.txt') as f:
    valid = [l.strip().split() for l in f] 

  with open('results/score/test.preds.txt') as f:
    test = [l.strip().split() for l in f] 

  for c in cfg:
    lines = valid if c[1] == 'valid' else test
    print(lines[c[5]])
    print(lines[c[6]])
    print('==============')

if sys.argv[1] == 'train':
  json_file = sys.argv[2] if len(sys.argv) > 2 else None
  train(restore=False, json_file=json_file)

elif sys.argv[1] == 'matrix':
  print_all = sys.argv[3] == 'P' if len(sys.argv) > 3 else False
  matrix('', int(sys.argv[2]), print_all=print_all)

elif sys.argv[1] == 'qmatrix':
  print_all = sys.argv[3] == 'P' if len(sys.argv) > 3 else False
  matrix('2', int(sys.argv[2]), print_all=print_all)

elif sys.argv[1] == 'quick':
  quick_train('valid', 4)

elif sys.argv[1] == 'test':
  test()

elif sys.argv[1] == 'pairs':
  test_pairs()

elif sys.argv[1] == 'allmatrices':
  for i in range(len(gcfg)):
    matrix('', i, verbose=False)
