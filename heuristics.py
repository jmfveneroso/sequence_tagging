import sys
import re

verbose = False
stats = False

def is_punctuation(text):
  return re.match("^[\,\;\:\-\"\(\)“”；\*]$", text)

def remove_honorifics(lines):
  honorifics = [
    'm.sc.', 'msc', 'sc.', 'sc', 'm.ed.', 'med', 'ed.', 'sc.nat.', 
    'scnat', 'nat.', 'nat', 'rer.nat.', 'rernat', 'i.r.', 'ir', 'pd.', 
    'pd', 'md.', 'md', 'b.sc.', 'bsc', 'bs.sc.', 'bssc', 'bs.', 'bs', 
    'ph.d.', 'phd', 'ph.', 'ph', 'ed.d.', 'edd', 'ms.', 'ms', 'hon.', 
    'ad.', 'ad', 'em.', 'em', 'apl.', 'apl', 'dipl.', 'dipl', 'professor',
    'prof.', 'prof', 'dphil', 'd.phil.', 'dphil.', 'tutor', 'associate', 'emeritus'
  ]
  
  counter = 0
  for i, _ in enumerate(lines):
    tkns = lines[i].split(' ')
    if len(tkns) < 3:
      continue

    if tkns[2] == 'I-PER' and tkns[0].lower() in honorifics:
      tkns[2] = 'O'
      lines[i] = ' '.join(tkns)
      counter += 1
     
      if verbose: 
        print(lines[i])
  if stats:
    print('Honorifics removed:', counter)
  return lines

def remove_numbers(lines):
  counter = 0
  for i, _ in enumerate(lines):
    tkns = lines[i].split(' ')
    if len(tkns) < 3:
      continue

    if tkns[2] == 'I-PER' and not re.match(r'[0-9]', tkns[0]) is None:
      tkns[2] = 'O'
      lines[i] = ' '.join(tkns)
      counter += 1
   
      if verbose:
        print(lines[i])
  if stats:
    print('Numbers removed:', counter)
  return lines

def remove_fringe_punctuation(lines):
  counter = 0
  for i, _ in enumerate(lines):
    tkns = lines[i].split(' ')
    if len(tkns) < 3:
      continue
  
    prev_tkn, next_tkn = 'O', 'O'
    if i-1 >= 0:
      prev_tkn = lines[i-1].split(' ')
      prev_tkn = 'O' if len(prev_tkn) < 3 else prev_tkn[2]
  
    if i+1 < len(lines):
      next_tkn = lines[i+1].split(' ')
      prev_tkn = 'O' if len(next_tkn) < 3 else next_tkn[2]
  
    if is_punctuation(tkns[0]) and tkns[2] == 'I-PER' and (prev_tkn == 'O' or next_tkn == 'O'):
      tkns[2] = 'O'
      lines[i] = ' '.join(tkns)
      counter += 1
      if verbose:
        print(lines[i])

  if stats:
    print('Fringe punctuation removed:', counter)
  return lines

def remove_single_names(lines):
  counter = 0
  for i, _ in enumerate(lines):
    tkns = lines[i].split(' ')
    if len(tkns) < 3:
      continue
  
    prev_tkn, next_tkn = 'O', 'O'
    if i-1 >= 0:
      prev_tkn = lines[i-1].split(' ')
      prev_tkn = 'O' if len(prev_tkn) < 3 else prev_tkn[2]
  
    if i+1 < len(lines):
      next_tkn = lines[i+1].split(' ')
      next_tkn = 'O' if len(next_tkn) < 3 else next_tkn[2]
 
    if tkns[2] == 'I-PER' and prev_tkn == 'O' and next_tkn == 'O':
      tkns[2] = 'O'
      lines[i] = ' '.join(tkns)
      counter += 1
      if verbose:
        print(lines[i])
  if stats:
    print('Single names removed:', counter)
  return lines

def remove_duplicates(lines):
  names = {}

  counter = 0
  i = 0
  while i < len(lines):
    tkns = lines[i].split(' ')
    if len(tkns) < 3:
      i += 1
      continue

    # Found name.
    if tkns[2] == 'I-PER':
      name = []
      for j in range(10):
        if i + j >= len(lines):
          break

        tkns_ = lines[i+j].split(' ')
        if len(tkns_) == 3 and tkns_[2] == 'I-PER':
          name.append(tkns_[0])
        else:
          key = ' '.join(name)
          if key in names and names[key] >= 2:
            for k in range(j):
              tkns_ = lines[i+k].split(' ')
              tkns_[2] = 'O'
              lines[i+k] = ' '.join(tkns_)
              if verbose:
                print(lines[i+k])
              counter += 1
          break

      key = ' '.join(name)
      if not key in names:
        names[key] = 0
      names[key] += 1
      i += j

    else:
      i += 1

  if stats:
    print('Duplicates removed:', counter)
  return lines

def filter_names(f_):
  with open(f_) as f:
    lines = [line.strip() for line in f]

    lines = remove_honorifics(lines)
    lines = remove_numbers(lines)
    lines = remove_fringe_punctuation(lines)
    lines = remove_single_names(lines)
    lines = remove_duplicates(lines)

    if not stats:
      for l in lines:
        print(l)

filter_names(sys.argv[1])
