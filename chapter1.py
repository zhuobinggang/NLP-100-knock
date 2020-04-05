import typing
from string import Template

def question0(text: str) -> str:
  print(text[::-1])

def question1(text):
  ids = [1,3,5,7]
  print(''.join([text[i:i+1] for i in ids]))

def bi_gram(sentence: typing.List) -> typing.List[str]:
  return [sentence[i:i+2] for i in range(len(sentence) - 1)]

def bi_words(sentence: str) -> typing.List[str]:
  return bi_gram(sentence.split())

def question5():
  it = 'I am an NLPer'
  print(bi_gram(it))
  print(bi_words(it))

def question6():
  x = set(n_chars('paraparaparadise'))
  y = set(n_chars('paragraph'))
  print(x.union(y))
  print(x.intersection(y))
  print(x.difference(y))

def question7(x=12, y="気温", z=22.4):
  s = Template('$x時の$yは$z')
  print(s.substitute(x=x, y=y, z=z))
  #print(s.substitute(x, y, z))
   
# question0('abcd egg')

question1('abcdefghijklmn')
