import json, gzip
import re
import requests
from string import Template

path = 'about-english.txt'

def read_lines(path):
  with open(path) as f:
    return f.readlines()


def q20():
  with gzip.open('jawiki-country.json.gz', mode='rt') as f:
    for line in filter(lambda x: json.loads(x)['title'] == "イギリス" ,f.readlines()):
      print(json.loads(line)['text'], end='')

def q21():
  for line in filter(lambda line: 'Category' in line, read_lines(path)):
    print(line, end='')

def q22():
  with open(path) as f:
    for line in filter(lambda line: 'Category' in line, f.readlines()):
      print(re.sub('\|?\*?\]\]$' ,'' ,re.sub('^\[\[Category:', '', line)))
      
def q23():
  for l in read_lines(path):
    if re.search('^=+', l):
      level = len(re.search('^=+', l).group(0))
      name = re.sub('[\s\n]', '', re.sub('=+$','',re.sub('^=+', '', l)))
      print(Template('$name: $level').substitute(name=name, level=level))

def q24():
  for l in read_lines(path):
    if 'ファイル' in l:
      print(re.sub('\|.*','',re.sub('.*ファイル:','', l)), end='')

def template_line_filtered():
  return filter(lambda l: re.search('\s=\s', l),read_lines(path))

def q25():
  result = {}
  for l in template_line_filtered():
    key = re.sub('\s.*' ,'',re.sub('^\|','', l))
    val = re.sub('.*\s=\s','', l)
    print(key)
    # result[key] = val

def q26():
  result = {}
  for l in template_line_filtered():
    key = re.sub('\s.*' ,'',re.sub('^\|','', l))
    val = re.sub('.*\s=\s','', l).replace('\'', '')
    result[key] = val

def q27():
  keys = []
  vals = []
  for l in template_line_filtered():
    keys.append(re.sub('\s.*' ,'',re.sub('^\|','', l)))
    vals.append(re.sub('(\[\[|\]\])','',re.sub('.*\s=\s','', l).replace('\'', '')))
  return (keys, vals)

def q28():
  keys, vals = q27()
  vals = list(map(lambda x: re.sub('(\{\{|\}\})', '', re.sub('<.*>','',x)), vals))
  result = {}
  for k,v in zip(keys, vals):
    result[k] = v
  return result

def get_resp():
  filename = q28()['国旗画像']
  r = requests.get(Template('https://en.wikipedia.org/w/api.php?action=query&format=json&prop=imageinfo&titles=File%3A$file&iiprop=url&iilimit=50&iiend=2007-12-31T23%3A59%3A59.000Z').substitute(file=filename))
  return r.text

def q29():
  res = get_resp()
  for url in map(lambda x: re.sub('(url|\"|\:)', '',x), re.findall('\"url\":\".*?\"', res)):
    print(url)

q29()