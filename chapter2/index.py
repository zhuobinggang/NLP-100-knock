import math
from string import Template

path = 'hightemp.txt'

def q10():
  return sum(1 for line in open(path))

def q11():
  with open(path) as f:
    print(f.read().replace('\t', ' '), end='')

def q12():
  with open(path) as f:
    col1_and_2 = [(line.split('\t')[0], line.split('\t')[1]) for line in f.readlines()]

  with open('col1.txt', mode='w') as f1, open('col2.txt', mode='w') as f2:
   for col1, col2 in col1_and_2:
     f1.write(col1 + '\n')
     f2.write(col2 + '\n')
    # print([col for col in map(list, zip(*cols))])

def q13():
  with open('col1.txt') as f1, open('col2.txt') as f2:
    for col1, col2 in zip(f1.readlines(), f2.readlines()):
      print((col1 + '\t' + col2).replace('\n', ''))

def q14(num: int):
  with open(path) as f:
    for line in f.readlines()[0: num]:
      print(line, end='')

def q15(num: int):
  with open(path) as f:
    for line in f.readlines()[-num:]:
      print(line, end='')


def q16(num: int, suffix = 'ax'):
  with open(path) as f:
    lines = [l for l in f.readlines()]
    line_num_average = math.ceil(len(lines) / num)
    for index, start in enumerate(list(range(0,len(lines), line_num_average))):
      content_to_write = lines[start: start + line_num_average]
      filename = Template('$suffix$index')
      with open(filename.substitute(suffix=suffix, index=(index + 1)), mode='w') as f:
        for line in content_to_write:
          f.write(line)


# [x] 1. To check whether the output of sort in py is the same as sort command
# [x] 2. To check how to use uniq command
# [x] 3. Just put all lines into the set and output the set
def q17():
  with open('col1.txt') as f:
    myset = set(f.readlines())
  for item in myset:
    print(item, end='')


# [x] sort using the command
# [x] sort using py
def q18():
  with open(path) as f:
    for line in sorted(f.readlines(), key=lambda line: -1 * line.split()[2]):
      print(line, end='')


def q19():
  with open(path) as f:
    first_col = list(map(lambda line: line.split()[0],f.readlines()))
    item_count_tuples = list(map(lambda item: (item, first_col.count(item)),set(first_col)))
    for item,count in sorted(item_count_tuples, key=lambda keycount: -1 * keycount[1]):
      print(item)

q19()