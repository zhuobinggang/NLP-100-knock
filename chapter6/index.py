import re
# from nltk.stem import PorterStemmer
# ps = PorterStemmer()
import xml.etree.ElementTree as ET
from string import Template

# path = 'nlp.txt'
path = 'q50.input'
xml_path = 'nlp.txt.xml'

def not_empty_line(line):
  return not re.match('^$', line)

def q50():
  # 1. split with regexp, then output for each sentence 
  result = []
  with open(path) as f:
    for l in filter(not_empty_line, f.readlines()):
      result += re.sub('([\.\;\:\?\!])\s([A-Z])', lambda match: match.group(1) + '\n' + match.group(2), l).split('\n')
  return filter(not_empty_line, result)


def q51():
  words = []
  for sentence in q50():
    words += list(map(lambda x: re.sub('[\.\;\:\?\!\,]', '', x), sentence.split()))
  return words
    
def q52():
  for w in q51():
    print(w, end='\t')
    print(ps.stem(w))

def q53():
  # root -> sentences -> sentence -> tokens -> token -> word
  for word in ET.parse('q50.input.xml').iter("word"):
    print(word.text)

def q54():
  # word: root -> sentences -> sentence -> tokens -> token -> word
  # lemma: root -> sentences -> sentence -> tokens -> token -> lemma
  # 品詞: root -> sentences -> sentence -> tokens -> token -> POS
  for token in ET.parse('nlp.txt.xml').iter("token"):
    print(Template('$word\t$lemma\t$POS').substitute(word=token.find('word').text, lemma=token.find('lemma').text, POS=token.find('POS').text))
    
def q55():
  for token in filter(lambda x: x.find('NER').text == 'PERSON' , ET.parse('nlp.txt.xml').iter("token")):
    print(Template('$word\t$lemma\t$POS').substitute(word=token.find('word').text, lemma=token.find('lemma').text, POS=token.find('POS').text))

class Sentence:
  def __init__(self, tokens):
    self.tokens = tokens 
  def __str__(self):
    return ' '.join([t.word for t in self.tokens])

class Token:
  def __init__(self, word):
    self.word = word

def sentences_from_root(root):
  return next(root.iter('sentences')).iter("sentence")


def q56():
  # 1. for each coreference, attach representative mention to the rest mentions
  # 2. then output all sentence

  root = ET.parse(xml_path)

  # Build all sentences from xml
  sentences: [Sentence] = []
  for sentence in sentences_from_root(root):
    tokens: [Token] = []
    for token in sentence.iter("token"):
      tokens.append(Token(token.find('word').text))
    sentences.append(Sentence(tokens))

  # attach representative mention to the other mentions
  for coreference in next(root.iter('coreference')).findall('coreference'):
    mentions = coreference.iter('mention')
    representative_mention = next(mentions)
    for mention in mentions:
      tokens = sentences[int(mention.find('sentence').text) - 1].tokens
      tokens[int(mention.find('start').text) - 1].word = '(' + tokens[int(mention.find('start').text) - 1].word
      tokens[int(mention.find('end').text) - 2].word += ')[' + representative_mention.find('text').text + ']'

  for s in sentences:
    print(s)

def deps_to_paths(deps):
  return '\n'.join(map(lambda x: x.find('dependent').text + ' -> ' + x.find('governor').text, filter(lambda x: re.match('\W',x.find('dependent').text) == None and re.match('\W',x.find('governor').text) == None, deps))) + '\n'

def all_collapsed_dependencies(sentence):
   return next(x for x in sentence.iter('dependencies') if x.attrib['type'] == 'collapsed-dependencies').findall('dep')

def q57():
  # sentence -> <dependencies type="collapsed-dependencies"> -> dep 
  root = ET.parse(xml_path)
  for s in sentences_from_root(root):
    result = ''
    result += 'digraph G {\n'
    result += deps_to_paths(all_collapsed_dependencies(s)) 
    result += '}\n'
    print(result)

def nsubjs_and_dobjs(deps):
  nsubjs = []
  dobjs = []
  for dep in deps:
    if dep.attrib['type'] == 'nsubj':
      nsubjs.append(dep)
    elif dep.attrib['type'] == 'dobj':
      dobjs.append(dep)
  return (nsubjs, dobjs)


def q58():
  # 述語: governer is the same, in two dependencies, in which types are 'dobj' & 'nsubj' independently
  # 1. for all nsubj deps, if there is a dobj share the same governor
  # 2. save (juchugo = nsubj.governor, syugo = nsubj.dependent, mokutekigo = dobj.dependent)
  root = ET.parse(xml_path)
  for s in sentences_from_root(root):
    all_nsubj_deps, all_dobj_deps = nsubjs_and_dobjs(all_collapsed_dependencies(s)) 
    # print(len(all_nsubj_deps), end=', ')
    # print(len(all_dobj_deps))
    for nsubj in all_nsubj_deps:
      for dobj in all_dobj_deps:
        if nsubj.find('governor').text == dobj.find('governor').text:
          print(Template('$syugo\t$juchugo\t$mokutekigo').substitute(juchugo=nsubj.find('governor').text, syugo=nsubj.find('dependent').text, mokutekigo=dobj.find('dependent').text))

class Tree:
  def __init__(self, name, is_terminal=False, val='', trees=[], rest=''):
    self.name = name
    self.is_terminal = is_terminal
    self.val = val
    self.trees = trees
    self.rest = rest
  def get_val(self):
    if self.is_terminal:
      return self.val
    else:
      return ' '.join(map(lambda x: x.get_val(), self.trees))
  def __str__(self):
    return self.name + ' ' + self.rest

def all_sub_tree_raw_strs(raw_str):
  raw_str = raw_str[1:-1] if re.match('^\(', raw_str) else raw_str
  # 1. encounter first (, start recording, encounter pair ), stop recording and save to results
  results = []
  recording = False
  record = ''
  stack = 0
  for index in range(0, len(raw_str)):
    record += raw_str[index] if recording else ''
    if raw_str[index] == '(':
      if stack == 0:
        recording = True
        record = '('
      stack += 1
    elif raw_str[index] == ')':
      stack -= 1
      if stack == 0:
        results.append(record)
        recording = False
        record = ''
  return results

def name_and_rest(raw_str):
  raw_str = raw_str[1:-1] if re.match('^\(', raw_str) else raw_str
  first_space = raw_str.index(' ')
  return (raw_str[0:first_space], raw_str[first_space + 1:])


def build_parse_tree(raw_str):
    name, rest = name_and_rest(raw_str)
    sub_strs = all_sub_tree_raw_strs(raw_str)
    if len(sub_strs) < 1:
      return Tree(name=name, is_terminal=True, val=rest, rest=rest)
    else:
      return Tree(name=name, is_terminal=False, trees=list(map(lambda x: build_parse_tree(x), sub_strs)), rest=rest)

def all_NP_tree(tree: Tree):
  result = []
  if tree.name == 'NP':
    result.append(tree)
  for child in tree.trees:
    result += all_NP_tree(child)
  return result

def q59():
  # Iterate all nodes of the tree, if the node's name is "NP", iterate and sum up all vals of subnodes. 
  for s in sentences_from_root(ET.parse(xml_path)):
    root = build_parse_tree(s.find('parse').text) 
    for tree in all_NP_tree(root):
      print(tree.get_val())
    print('---')

q59()
