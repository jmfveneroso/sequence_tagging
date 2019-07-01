import sys
import numpy as np

def f(a, p, r):
  x = a * (1 / float(r)) + (1-a) * (1 / float(p))
  return float(1 / x)

values = []

def key_fn(val): 
    return val[2] 

for l in sys.stdin:
  print(l)
  l = l.strip().split()

  _, p2, r2, _, _, p1, r1, _ = l
  if float(p2) < 0.1: continue

  a = 0.5
  x = [float(p1), float(r1)]
  x.append(float('%.4f'%f(a, p1, r1)))
  x.append(0)
  x += [float(p2), float(r2)]
  x.append(float('%.4f'%f(a, p2, r2)))
  x.append(0)

  values.append(x)

print(values)
values.sort(key=key_fn, reverse=True)
# print(values)

for v in values:
  print('\t'.join([str(v_) for v_ in v]))

values = np.array(values[:5])

mean = np.mean(values, axis=0)
std = np.std(values, axis=0)

print('\t'.join(['%.4f'%m + ' (' + '%.4f' % std[i] + ')' for i, m in enumerate(mean)]))
print('\t'.join(['%.4f'%m for i, m in enumerate(mean)]) + '\t' + '\t'.join(['%.4f'%m for i, m in enumerate(std)]))
