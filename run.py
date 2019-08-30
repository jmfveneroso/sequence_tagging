import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import tensorflow as tf
from optparse import OptionParser
from pathlib import Path
from model.estimator import Estimator
from model.hmm import HiddenMarkov, load_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable debug logs Tensorflow.
tf.logging.set_verbosity(tf.logging.ERROR)

parser = OptionParser()
parser.add_option("-j", "--json", dest="json_file")
parser.add_option("-s", "--small", dest="small", action="store_true")
(options, args) = parser.parse_args()
options = vars(options)

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

  if not json_file is None:
    estimator.load_params_from_file(json_file)
  estimator.train()
  estimator.test()

elif sys.argv[1] == 'test':
  json_file = options['json_file']

  estimator = Estimator()

  if not json_file is None:
    estimator.load_params_from_file(json_file)
  estimator.test()

elif sys.argv[1] == 'hmm':
  start_time = time.time()
  timesteps = int(sys.argv[2])
  naive_bayes = timesteps == 0
  if naive_bayes:
    timesteps = 1
  
  print('Fitting...')
  # X, Y, _ = load_dataset('data/conll2003_person/train')
  X, Y, _ = load_dataset('data/ner_on_html/train')
  hmm = HiddenMarkov(
    timesteps, 
    naive_bayes=naive_bayes,
    use_features=True,
    self_train=True
  )
  hmm.fit(X, Y)

  for name in ['train', 'valid', 'test']:
    print('Predicting ' + name)
    # x, t, w = load_dataset('data/conll2003_person/' + name)
    x, t, w = load_dataset('data/ner_on_html/' + name)
    p = hmm.predict(x)

    t = [[['O', 'B-PER', 'I-PER'][t__] for t__ in t_] for t_ in t]
    p = [[['O', 'B-PER', 'I-PER'][p__] for p__ in p_] for p_ in p]

    with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
      for words, preds, tags in zip(w, p, t):
        f.write(b'\n')
        for word, pred, tag in zip(words, preds, tags):
          f.write(' '.join([word, tag, pred]).encode() + b'\n')

  print('Elapsed time: %.4f' % (time.time() - start_time))
