import tensorflow as tf
import numpy as np
from pathlib import Path
from model.data_loader import DL
import tensorflow_hub as hub
import time

def precalculate_elmo():
  start_time = time.time()
  
  DL().set_params({
    'datadir': 'data/small_dataset'
  })
  elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
  
  # embs= np.zeros((302812, 1024))
  # embs= np.zeros((204568, 1024))
  # embs= np.zeros((98244, 1024))
  embs= np.zeros((0, 1024))
  
  with tf.Session() as sess:
    sess.run([tf.initializers.global_variables(), tf.tables_initializer()])
  
    sentences = []
    n_words = []
    uids_ = []
  
    for f in ['train', 'valid', 'test']:
      print(f)
      iterator = DL().input_fn(f, training=False).make_initializable_iterator()
      next_el = iterator.get_next()
      sess.run(iterator.initializer)
  
      z = 0
      u_ = 0
      while True:
        try:
          features, labels = sess.run(next_el)
          (words, uids, nwords), (_, _) = features
  
          words = [w.decode("utf-8") for w in words[0]]
          sentences.append(words)
          n_words.append(nwords[0])
          uids_.append(uids[0])
        except tf.errors.OutOfRangeError: 
          break
      
    print('finished loading')
    print(len(uids_))
    
    max_len = max([len(s) for s in sentences])
    for i, s in enumerate(sentences):
        sentences[i] = s + [''] * (max_len - len(s))
  
    size = 100
    for q in range(0, len(sentences), size):
        print('doing', q)
        s_ = sentences[q:q+size]
        n_ = n_words[q:q+size]
        u_ = uids_[q:q+size]
        print(u_)
        
        embeddings = elmo(
          inputs={
            "tokens": s_,
            "sequence_len": n_
          },
          signature="tokens",
          as_dict=True
        )["elmo"]
  
        e = sess.run(embeddings)
  
        for i, _ in enumerate(e):
          for j in range(n_[i]): 
            uid = u_[i][j]
            embs[uid,:] = e[i,j,:]
        print('done with', q)
  
    print('Elapsed time: %.4f' % (time.time() - start_time))
      
    np.savez_compressed('data/validtest_elmo_embeddings.npz', embeddings=embs)

def concat_pretrained():
  first = np.load('data/train_elmo_embeddings.npz')['embeddings']
  second = np.load('data/validtest_elmo_embeddings.npz')['embeddings']

  embs = np.vstack([first, second])
  np.savez_compressed('data/elmo_embeddings.npz', embeddings=embs)
  print('Done')
