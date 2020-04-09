import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
import matplotlib as mpl
import math

import numpy as np

path = 'neko.txt.cut.mecab'
#path = 'neko.txt.mecab'

def q30():
  the_list = []
  with open(path) as f:
    for l in filter(lambda x: 'EOS' not in x ,f.readlines()):
      if('\t' not in l):
        continue
      item = {}
      item['surface'] = l.split('\t')[0]
      others = l.split('\t')[1].split(',')
      item['base'] = others[-3]
      item['pos'] =  others[0]
      item['pos1'] = others[1]
      the_list.append(item)
  return the_list

def q31():
  for dic in filter(lambda x: x['pos'] == '動詞',q30()):
    print(dic['surface'])

def q32():
  for dic in filter(lambda x: x['pos'] == '動詞',q30()):
    print(dic['base'])

def q33():
  for dic in filter(lambda x: x['pos1'] == 'サ変接続',q30()):
    print(dic['base'])

def q34():
  # 1. for index of each 'no'
  # 2. check if index - 1 && index + 1 == Noun
  # 3. print [index - 1] + no + [index + 1]
  dics = q30()
  for index, dic in enumerate(dics):
    if(dic['base'] == 'の' and dics[index - 1]['pos'] == '名詞' and dics[index - 1]['pos'] == '名詞'):
      print(dics[index - 1]['base'], end=' ')
      print(dic['base'], end=' ')
      print(dics[index + 1]['base'])

def q35():
  # 1. iterate the list for each noun if is not recording then start recording && empty recorded list
  # 2. for each non-noun stop recording & append to the matched list if is recording 
  recording = False
  recorded = []
  matched = []
  for dic in q30():
    if dic['pos'] == '名詞':
      if not recording: 
        recording = True
        recorded = [dic['base']] 
      else: 
        recorded.append(dic['base'])
    elif recording:
      recording = False
      matched.append(recorded.copy())
      recorded.clear() 
  if len(recorded) != 0:
    matched.append(recorded.copy())
  for l in matched:
    print(l)

def q36():
  keys = set([dic['base'] for dic in q30()])
  key_count_map = {}.fromkeys(keys, 0)
  for dic in q30():
    key_count_map[dic['base']] += 1
  result = list(sorted(key_count_map.items(), key=lambda x: x[1], reverse=True))
  #print(result)
  return result

def q37():
  key_count_pair = q36()[0:10]
  keys,vals = list(zip(*key_count_pair))
  mpl.rcParams['font.family'] = 'AppleGothic'
  plt.bar(keys, vals)
  plt.show()

def q38():
  key_count_pair = q36()
  xs = list(map(lambda x: x[1], key_count_pair))
  print(xs)
  plt.hist(xs)
  plt.show()

def q39():
  key_count_pair = list(q36())
  keys,vals = list(zip(*key_count_pair))
  mpl.rcParams['font.family'] = 'AppleGothic'
  logx = [math.log(i+1) for i in range(len(keys))]
  logy = [math.log(i+1) for i in vals]
  plt.scatter(logx, logy)
  plt.show()

q39()
