import sys

with open(sys.argv[1]) as f:
  for l in f:
    line = l.split()
    if len(line) < 3:
      continue

    error = (line[1] != line[2])
    if error:
      print('==============')
    print(l.strip())
    if error:
      print('==============')
