import matplotlib
matplotlib.use('Agg')
from model.estimator import Estimator
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tensorflow as tf
from optparse import OptionParser
from PIL import Image

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

parser = OptionParser()
parser.add_option("-j", "--json", dest="json_file")
parser.add_option("-s", "--small", dest="small", action="store_true")
(options, args) = parser.parse_args()
options = vars(options)

def matrix(example, print_all=False, verbose=True):
  global gcfg
  cfg = gcfg[example]

  estimator = Estimator()
  words, alphas_, preds, labs = estimator.get_alphas(cfg[1], cfg[2])
  words = [w.decode("utf-8") for w in words]

  first = cfg[3] 
  second = cfg[4]
  p1 = preds[first].decode("utf-8")
  p2 = preds[second].decode("utf-8")
  label = labs[first].decode("utf-8")
  tick = 'Y' if p1 == p2 == label else 'N'
  msg = '[%s] %s (%s) %s (%s) should be %s' % (tick, words[first], p1, words[second], p2, label)
  print(msg)

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
    
    fig = plt.figure(figsize=(10.0, 10.0), dpi=100)
    plt.title(msg)
    plt.xticks(range(0,end), words[start:end], rotation='vertical') 
    plt.yticks(range(0,end), words[start:end])
        
    plt.imshow(alphas_[start:end,start:end])
    plt.savefig('figures/' + str(example) + '.png', dpi=100)

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
  small = not options['small'] is None
  json_file = options['json_file']

  estimator = Estimator()

  if not json_file is None:
    estimator.load_params_from_file(json_file)

  if small:
    estimator.set_dataset_params({
      'datadir': 'data/small_dataset'
    })

  # Pre train. 
  estimator.set_dataset_params({
    'fulldoc': False,
    'batch_size': 1,
    'epochs': 1
  })
  estimator.train()

  if not json_file is None:
    estimator.load_params_from_file(json_file)
  estimator.train(restore=True)

  estimator.test()

if sys.argv[1] == 'params':
  json_file = options['json_file']
  estimator = Estimator()
  if not json_file is None:
    estimator.load_params_from_file(json_file)

elif sys.argv[1] == 'test':
  estimator = Estimator()
  estimator.test()

elif sys.argv[1] == 'pairs':
  test_pairs()

elif sys.argv[1] == 'print_matrices':
  for i in range(len(gcfg)):
    matrix(i, verbose=False)

  # Concat images.
  filenames = ['figures/' + str(i) + '.png' for i in range(len(gcfg))]
  images = list(map(Image.open, filenames))

  widths, heights = zip(*(i.size for i in images))
  total_width = max(widths)
  total_height = sum(heights)

  new_im = Image.new('RGB', (total_width, total_height))
  y_offset = 0
  for im in images:
    new_im.paste(im, (0, y_offset))
    y_offset += im.size[1]

  new_im.save('figures/all.png', 'PNG')
