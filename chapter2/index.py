path = 'hightemp.txt'

def q10():
  return sum(1 for line in open(path))

def q11():
  with open(path) as f:
    print(f.read().replace('\t', ' '), end='')

q11()
