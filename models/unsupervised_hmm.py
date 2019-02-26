import sys
import numpy as np

states = np.array(['H', 'C'])
features = np.array(['1', '2', '3'])

start = [.5, .5]
end   = [1, 1]

transition_mat = np.array([
  [.6, .4],
  [.4, .6]
])

emission_mat = np.array([
  [.2, .5],
  [.4, .4],
  [.4, .1]
])

observations = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1])

def forward_algorithm(observations):
  global transition_mat, transition_mat
  alphas = np.zeros((len(observations), len(states)))

  alpha = None
  for i, o in enumerate(observations):
    idx = o - 1
    if alpha is None:
      alpha = start * emission_mat[idx]
      alphas[i] = alpha
      continue

    alpha = np.matmul(alpha, transition_mat)
    alpha *= emission_mat[idx]
    alphas[i] = alpha
  return alphas, np.matmul(alpha, np.transpose(end))

def backward_algorithm(observations):
  global transition_mat, transition_mat
  betas = np.zeros((len(observations), len(states)))

  beta = end
  betas[len(observations) - 1] = beta
  for i, o in enumerate(reversed(observations)):
    idx = o - 1
    beta *= emission_mat[idx]
    if i < len(observations) - 1:
      beta = np.matmul(beta, np.transpose(transition_mat))
      betas[len(observations) - i - 2] = beta
  return betas, np.matmul(beta, np.transpose(start))

def viterbi(observations):
  global states
  backprobs = np.zeros((len(observations), len(states)))
  backpointers = np.zeros((len(observations), len(states)))

  alpha = None
  for i, o in enumerate(observations):
    idx = o - 1
    if alpha is None:
      alpha = start * emission_mat[idx]
      backprobs[i] = alpha
      continue

    alpha_mat = transition_mat * emission_mat[idx]
    alpha_mat = np.transpose(alpha_mat) * alpha
    alpha = np.amax(alpha_mat, axis=1)
    pointers = np.argmax(alpha_mat, axis=1)

    backprobs[i] = np.amax(alpha_mat, axis=1)
    backpointers[i] = np.argmax(alpha_mat, axis=1)

  last_state = np.argmax(backprobs[len(observations) - 1])
  res = [states[last_state]]
  for i in range(0, len(observations) - 1):
    last_state = int(backpointers[len(observations) - i - 1][last_state])
    # print last_state
    res.append(states[last_state])
    
  # print backprobs
  # print backpointers
  res.reverse()
  return res

# print forward_algorithm(observations)

def forward_backward_algorithm(observations):
  global emission_mat, transition_mat
  for i in range(0, 100):
    alphas, n = forward_algorithm(observations)
    betas,  n = backward_algorithm(observations)

    # Transition probs.
    numerator = np.matmul(np.transpose(alphas), betas) * transition_mat
    denominator = np.sum(numerator, axis=1)
    new_transition_probs = (numerator.T / denominator).T

    # Emission probs.
    unary = np.zeros((len(observations), len(features)))
    for i, o in enumerate(observations):
      idx = o - 1
      unary[i][idx] = 1

    numerator = alphas.T * betas.T
    denominator = np.sum(numerator, axis=1)
    numerator = np.matmul(numerator, unary)
    new_emission_probs = numerator.T / denominator

    # print np.round(transition_mat, 4)
    # print np.round(new_transition_probs, 4)
    # print np.round(emission_mat, 4)
    # print np.round(new_emission_probs, 4)

    transition_mat = new_transition_probs
    emission_mat = new_emission_probs
  print( np.round(transition_mat, 4))
  print( np.round(emission_mat, 4))

# forward_backward_algorithm(observations)
# print [str(x) for x in observations]
# print viterbi(observations)

























import numpy as np
import re
from pathlib import Path

def load_raw_dataset(f):
  with open(f, 'r') as f:
    data = f.read().strip()

    sentences = data.split('\n\n')
    sentences = [s for s in sentences if not s.startswith('-DOCSTART-')]
    X = [[t.split(' ') for t in s.split('\n') if len(s) > 0] for s in sentences]
    Y = []
    T = []
    for i, s in enumerate(X):
      tkns, labels = [], []
      for j, t in enumerate(s):
        l = ['O', 'I-PER'].index(t[1])
        labels.append(l)
        tkns.append(t[0])
        X[i][j] = [X[i][j][0]] + X[i][j][2:]

      Y.append(labels)
      T.append(tkns)

    return X, Y, T

class HiddenMarkov:
  def __init__(self):
    self.time_steps   = 1

    self.num_labels   = 2
    self.num_features = 11
    self.num_states = self.num_labels ** self.time_steps
    self.transition_mat = np.ones((self.num_states, self.num_labels))
    self.start = np.zeros((self.num_states, 1))
    self.start[0,:] = 1 # All previous states are label O ("other").
    self.end = np.ones((self.num_states, 1)) # All ending states are equally probable.

    self.feature_counts = []
    for i in range(self.num_features):
      self.feature_counts.append([])
      for j in range(self.num_labels):
        self.feature_counts[i].append({'$UNK': 1})

  def idx_to_states(self, idx):
    states = []
    multiplier = self.num_labels ** (self.time_steps - 1)
    for i in range(self.time_steps):
      states.append(int(idx) // int(multiplier))
      idx %= multiplier
      multiplier /= self.num_labels
    return states 
  
  def states_to_idx(self, states):
    if len(states) < self.time_steps:
      raise Exception('Wrong states length.')
  
    acc = 0
    multiplier = 1
    for s in reversed(states):
      acc += int(multiplier) * int(s)
      multiplier *= self.num_labels
    return acc

  def train_features(self, X, Y, which_features=[]):
    if len(which_features) != self.num_features:
      which_features = [0] * self.num_features

    label_count = np.ones((self.num_labels))
    for i in range(len(Y)):
      for j in range(len(Y[i])):
        label_count += Y[i][j]
        y = Y[i][j]
  
        f = X[i][j][1:1+self.num_features]
        for k in range(self.num_features):
          if which_features[k] == 0:
            continue

          key = ''
          if k < len(f):
            key = f[k]

          if not key in self.feature_counts[k][y]:
            self.feature_counts[k][y][key] = 0
          self.feature_counts[k][y][key] += 1
 
    # Consolidate vocabularies. 
    feature_maps = []
    for i in range(self.num_features):
      feature_maps.append({})
      for j in range(self.num_labels):
        for k in self.feature_counts[i][j]:
          feature_maps[i][k] = True

    for i in range(self.num_features):
      if which_features[i] == 0:
        continue

      for j in range(self.num_labels):
        for k in feature_maps[i]:
          if not k in self.feature_counts[i][j]:
            self.feature_counts[i][j][k] = 1

    for i in range(self.num_features):
      if which_features[i] == 0:
        continue

      for j in range(self.num_labels):
        total_count = sum([self.feature_counts[i][j][k] for k in self.feature_counts[i][j]])
        for k in self.feature_counts[i][j]:
          self.feature_counts[i][j][k] /= float(total_count)

  def train_transitions(self, X, Y):
    for i in range(len(Y)):
      states = [0] * self.time_steps
      for j in range(len(Y[i])):
        y = Y[i][j]
        idx = self.states_to_idx(states)

        self.transition_mat[idx,y] += 1
        states.pop(0) 
        states.append(y) 
  
    self.transition_mat /= np.expand_dims(np.sum(self.transition_mat, axis=1), axis=1)
    self.transition_mat = np.nan_to_num(self.transition_mat)

  def fit(self, X, Y):
    which_features = [1] * self.num_features 
    self.train_features(X, Y, which_features)
    self.train_transitions(X, Y)

  def viterbi(self, X):
    pointers = np.zeros((len(X), self.num_states), dtype=int)
  
    state_probs = self.start 
    for i in range(len(X)):
      emission = np.ones(self.num_labels)
  
      f = X[i][1:1+self.num_features]
      for k in range(self.num_features):
        for y in range(self.num_labels):
          key = ''
          if k < len(f):
            key = f[k]

          if key in self.feature_counts[k][y]: 
            emission[y] *= self.feature_counts[k][y][key]
          else:
            emission[y] *= self.feature_counts[k][y]['$UNK']
      emission[emission == 1] = 0

      p = state_probs * self.transition_mat * emission

      state_probs = np.zeros((self.num_states, 1))
      for s in range(self.num_states):
        for l in range(self.num_labels):
          states = self.idx_to_states(s)
          states.pop(0)
          states.append(l)
          idx = self.states_to_idx(states)

          if p[s,l] > state_probs[idx,0]:
            pointers[i,idx] = s
            state_probs[idx,0] = p[s,l]

    idx = np.argmax(state_probs)
    labels = [] 
    for i in reversed(range(len(X))):
      states = self.idx_to_states(idx)
      labels.append(states[-1])
      idx = pointers[i,idx]
    labels = list(reversed(labels))

    return labels

  def predict(self, X):
    y = []
    for i in range(len(X)):
      labels = self.viterbi(X[i])
      y.append(labels) 
    return y

if __name__ == '__main__':
  print('Fitting...')
  X, Y, _ = load_raw_dataset('data/ner_on_html/train')
  hmm = HiddenMarkov()
  hmm.fit(X, Y)

  for name in ['train', 'valid', 'test']:
    print('Predicting ' + name)
    x, t, w = load_raw_dataset('data/ner_on_html/' + name)
    p = hmm.predict(x)

    t = [[['O', 'I-PER'][t__] for t__ in t_] for t_ in t]
    p = [[['O', 'I-PER'][p__] for p__ in p_] for p_ in p]

    with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
      for words, preds, tags in zip(w, p, t):
        f.write(b'\n')
        for word, pred, tag in zip(words, preds, tags):
          f.write(' '.join([word, tag, pred]).encode() + b'\n')
