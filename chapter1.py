import typing
import random
from string import Template

def question0(text: str) -> str:
  return text[::-1]

def question1(text):
  return ''.join(text[::2])

def question2(text1, text2):
  return [a+b for a,b in zip(text1, text2)]

def question3(sentence: str) -> typing.List[int]:
  return [i for i in map(lambda  x: len(x.replace(',', '').replace('.','')) ,sentence.split())]

def question4(sentence: str):
  ids = [i-1 for i in [1, 5, 6, 7, 8, 9, 15, 16, 19]]
  dic = {}
  for i,s in enumerate(sentence.split()):
    if i in ids:
      dic[s[0]] = i + 1
    else:
      dic[s[0:2]] = i + 1
  return dic

def bi_gram(sentence: typing.List) -> typing.List[str]:
  return [sentence[i:i+2] for i in range(len(sentence) - 1)]

def bi_words(sentence: str) -> typing.List[str]:
  return bi_gram(sentence.split())

def question5():
  it = 'I am an NLPer'
  return (bi_gram(it), bi_words(it))

def question6():
  x = set(bi_gram('paraparaparadise'))
  y = set(bi_gram('paragraph'))
  return (x.union(y), x.intersection(y), x.difference(y))

def question7(x=12, y="気温", z=22.4):
  s = Template('$x時の$yは$z')
  return s.substitute(x=x, y=y, z=z)
  #print(s.substitute(x, y, z))

def question8(s: str):
  return ''.join([c if not c.islower() else chr(219 - ord(c)) for c in s])

def random_word(word: str):
  it = list(word)
  random.shuffle(it)
  return ''.join(it)

def question9(s: str):
  # return [word[1:-1] for word in s.split()]
  return [word[0] + random_word(word[1:-1]) + word[-1] if len(word) > 3 else word for word in s.split()]
   
# question0('abcd egg')
#question2('abc', 'ef')

# question3("Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.")

# question4("Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.")

# print(question8("Hi He Lied."))

# print(question9("I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."))

# print(question6())

