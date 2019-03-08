import sys
import numpy as np

def f(a, p, r):
  x = a * (1 / float(r)) + (1-a) * (1 / float(p))
  return float(1 / x)

values = []

for l in sys.stdin:
  l = l.strip().split()

  p1, r1, p2, r2 = l

  x = [float(p1), float(r1)]
  x.append(f(0.2, p1, r1)) 
  x.append(f(0.5, p1, r1)) 
  x.append(f(0.8, p1, r1)) 

  x += [float(p2), float(r2)]
  x.append(f(0.2, p2, r2)) 
  x.append(f(0.5, p2, r2)) 
  x.append(f(0.8, p2, r2)) 

  values.append(x)

values = np.array(values)

values = values[15:,:]

mean = np.mean(values, axis=0)
std = np.std(values, axis=0)

print('\t'.join(['%.4f'%m + ' (' + '%.4f' % std[i] + ')' for i, m in enumerate(mean)]))
