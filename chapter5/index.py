from string import Template
from typing import TypeVar
T = TypeVar('T')

# path = 'ee.txt'
# path = 'dd.cabocha'
# path = 'neko.cut.cabocha'
# path = 'q46.output'
path = 'q48.input'

class Chunk:
  def __init__(self, morphs, dst, srcs, id_in_sentence):
    self.morphs = morphs
    self.dst = dst
    self.srcs = srcs
    self.id_in_sentence = id_in_sentence
  def __str__(self):
    t = Template('morphs: $morphs, dst: $dst, srcs: $srcs')
    morphs = ','.join(map(lambda x: x.base, self.morphs)) 
    srcs = ','.join(self.srcs)
    return t.substitute(morphs=morphs, dst=self.dst, srcs=srcs)
  def text(self):
    return ''.join(map(lambda x: x.surface, self.morphs))
  def __eq__(self, other):
    return self.id_in_sentence == other.id_in_sentence

  

class Morph:
  def __init__(self, surface, base, pos, pos1):
    self.surface = surface
    self.base = base
    self.pos = pos
    self.pos1 = pos1 
  def __str__(self):
    return Template('surface: $surface,base: $base,pos: $pos,pos1: $pos1').substitute(surface=self.surface, base=self.base, pos=self.pos, pos1=self.pos1)

def morph(line: str):
  cols = line.split('\t')
  return Morph(
    surface=cols[0],
    base=cols[1].split(',')[-3],
    pos=cols[1].split(',')[0],
    pos1=cols[1].split(',')[1],
  )

def meaning_lines():
  with open(path) as f:
    return filter(lambda line: 'EOS' not in line and not line.startswith('*'),f.readlines()) 

def q40():
  morphs = []
  for line in meaning_lines():
    morphs.append(morph(line))
  return morphs

def splited_sentences(lines: [str]):
  sentences = []
  sentence = []
  for line in lines:
    if('EOS' in line):
      sentences.append(sentence.copy())
      sentence.clear()
    else:
      sentence.append(line)
  return sentences

def cabocha_lines():
  with open(path) as f:
    return f.readlines()

def chunks(sentence: [str]):
  dst = -2
  morphs = []
  chunks = []
  for index, line in enumerate(sentence):
    if line.startswith('*'):
      if len(morphs) > 0:
        chunks.append(Chunk(morphs=morphs.copy(), dst=dst, srcs=[], id_in_sentence= index))
      dst=int(line.split(' ')[2].split('D')[0])
      morphs.clear()
    else:
      morphs.append(morph(line))
  chunks.append(Chunk(morphs=morphs.copy(), dst=dst, srcs=[], id_in_sentence=len(sentence)))
  return chunks

def q41():
  sentences = splited_sentences(cabocha_lines())
  chunks_in_sentences = map(chunks, sentences)
  result = []
  for chunks_in_sentence in chunks_in_sentences:
    for index, c in enumerate(chunks_in_sentence):
      if(c.dst != '-1'):
        chunks_in_sentence[c.dst].srcs.append(str(index))
    result.append(chunks_in_sentence)
    # for c in chunks_in_sentence:
    #   print(c)
    # print()
  return result


def q42():
  chunks_in_sentences = q41()
  for chunks_in_sentence in chunks_in_sentences:
    for c in chunks_in_sentence:
      if(c.dst != -1):
        print(c.text() + chunks_in_sentence[c.dst].text())
    print('-----')

def chunk_include_noun(chunk: Chunk):
  return  len(list(filter(lambda x: x.pos == '名詞', chunk.morphs))) > 0

def chunk_include_verb(chunk: Chunk):
  return  len(list(filter(lambda x: x.pos == '動詞', chunk.morphs))) > 0

def dst_chunk(chunk, chunks_in_sentence):
  return chunks_in_sentence[chunk.dst]

def q43():
  # For each chunk in sentence, if include_noun and dst_chunk include_verb then
  # print chunk tab dst_chunk
  for chunks_in_sentence in q41():
    for chunk in chunks_in_sentence:
      if chunk_include_noun(chunk) and chunk_include_verb(dst_chunk(chunk, chunks_in_sentence)):
        print(chunk.text() + '\t' +  dst_chunk(chunk, chunks_in_sentence).text())

class Node:
  def __init__(self, chunk: Chunk, parent):
    self.children = []
    self.chunk = chunk
    self.parent = parent
  def __str__(self):
    return self.chunk.text()
  def dot(self):
    result = ''
    for node in self.children:
      result += node.chunk.text() + ' -> ' + self.chunk.text() + '\n'
      result += node.dot()
    return result
  def __eq__(self, other):
    if other == None:
      return False
    # print('__eq__' + str(self.chunk.id_in_sentence) + str(other.chunk.id_in_sentence))
    return self.chunk == other.chunk

def dot(node: Node):
  result = ''
  result += 'digraph G {\n'
  result += node.dot()
  result += '}\n'
  return result
  

def q44():
  # 1. foreach chunk in the sentence, create a node
  # 2. foreach chunk&node in the sentence, if the chunk have a dst != -1, add the node to the children array of dst_node, else set the node as root
  roots = []
  for chunks_in_sentence in q41():
    chunk_node_pairs = []
    root = None
    for chunk in chunks_in_sentence:
      chunk_node_pairs.append((chunk, Node(chunk, parent=None))) 
    for chunk, node in chunk_node_pairs:
      if chunk.dst != -1:
        chunk_node_pairs[chunk.dst][1].children.append(node)
        node.parent = chunk_node_pairs[chunk.dst][1]
      else:
        root = node
    if root == None:
      raise ValueError('There must be something wrong with the q44')
    else:
      roots.append(root)
  return roots

def first_verb_in_chunk(chunk: Chunk) -> Morph:
  for m in chunk.morphs:
    if m.pos == '動詞':
      return m

def all_nodes(root: Node) -> [Node]:
  result = [root]
  for child in root.children:
    for node in all_nodes(child):
      result.append(node)
  return result

def all_node_chunk_pairs(root: Node) -> [(Node, Chunk)]:
  result = [(root, root.chunk)]
  for child in root.children:
    for node in all_nodes(child):
      result.append((node, node.chunk))
  return result

def all_joshi_in_chunk(c: Chunk) -> [Morph]:
  return filter(lambda x: x.pos == '助詞',c.morphs)

def q45():
  # 1. for each chunk in the sentence, if include_verb then
  # 2. output base, then
  # 3. for each children chunk, output joshi
  for root in q44():
    for node in all_nodes(root):
      if chunk_include_verb(node.chunk):
        print(first_verb_in_chunk(node.chunk).base, end=' ')
        for c in node.children:
          for m in all_joshi_in_chunk(c.chunk):
            print(m.base, end=' ')
        print()
    print('------' + root.chunk.text())
        

def chunk_include_joshi(c: Chunk):
  return len(list(filter(lambda x: x.pos == '助詞',c.morphs))) > 0

def q46():
  # 1. for each chunk in the sentence, if include_verb then
  # 2. output base, then
  # 3. for each children chunk, output joshi and
  # 4. for each children chunk, if include_joshi then output chunk
  for root in q44():
    for node in all_nodes(root):
      if chunk_include_verb(node.chunk):
        print(first_verb_in_chunk(node.chunk).base, end=' ')
        for n in node.children:
          for m in all_joshi_in_chunk(n.chunk):
            print(m.base, end='\t')
        for n in node.children:
          if chunk_include_joshi(n.chunk):
            print(n.chunk.text(), end= '\t')
        print()
    #print('------' + root.chunk.text())

def q47():
  # 1. for each chunk in sentence, if first_morph.pos == '名詞' and first_morph.pos1 == 'first_morph.pos' and second_morph.base == 'を' and include_verb(dst_chunk) then
  # 2. verb = first_verb(dst_chunk(chunk)) and output predicate and
  # 3. for each child chunk of the dst_chunk, if not equal to the chunk, output joshi
  # 4. for each child chunk of the dst_chunk, if not equal to the chunk, output all
  for root in q44():
    for node, chunk in all_node_chunk_pairs(root):
      # print(chunk.morphs[0].pos, end=', ')
      # print(chunk.morphs[0].pos1, end=', ')
      # if len(chunk.morphs) > 1:
      #   print(chunk.morphs[1].base, end=', ')
      # print()
      if len(chunk.morphs) > 1 and chunk.morphs[0].pos == '名詞' and chunk.morphs[0].pos1 == 'サ変接続' and chunk.morphs[1].base == 'を' and chunk_include_verb(node.parent.chunk):
        verb = first_verb_in_chunk(node.parent.chunk)
        print(chunk.text() + verb.base, end='\t')
        joshi_chunk_pairs = []
        for node in node.parent.children:
          if node.chunk.id_in_sentence != chunk.id_in_sentence:
            reversed_joshis = list(reversed(list(all_joshi_in_chunk(node.chunk))))
            if len(reversed_joshis) > 0:
              joshi_chunk_pairs.append((reversed_joshis[0].base, node.chunk))
        
        sorted_joshi_chunk_pairs = sorted(joshi_chunk_pairs, key=lambda x: x[0])
        for joshi, _ in sorted_joshi_chunk_pairs:
          print(joshi, end='\t')
        for _, chunk in sorted_joshi_chunk_pairs:
          print(chunk.text(), end='\t')
        print()

def path_to_root(node: Node):
  if node == None:
    return ''
  else:
    return ' -> ' + node.chunk.text() + path_to_root(node.parent)
            
def q48():
  for root in q44():
    for node in filter(lambda x: chunk_include_noun(x.chunk), all_nodes(root)):
      print(node.chunk.text() + path_to_root(node.parent))

def is_connected(node1: Node, node2: Node):
  if node1 == None:
    return False
  elif node1.chunk.id_in_sentence == node2.chunk.id_in_sentence:
    return True
  else:
    return is_connected(node1.parent, node2)

def pairs(l: [T]) -> [(T, T)]:
  l = list(l)
  result = []
  for i in range(len(l)):
    for j in range(i+1, len(l)):
      result.append((l[i], l[j]))
  return result

def the_rest_path_node_to_node(x: Node, y: Node):
  if x.chunk.id_in_sentence == y.chunk.id_in_sentence:
    return ' -> ' + x.chunk.text()
  else:
    return ' -> ' + x.chunk.text() + the_rest_path_node_to_node(x.parent, y)

def path_node_to_node(x: Node, y: Node):
  return x.chunk.text() + the_rest_path_node_to_node(x.parent, y)

def node_ancestors(n: Node) -> [Node]:
  if n.parent == None:
    return []
  else:
    return [n.parent] + node_ancestors(n.parent)

def contain_node(l, n):
  for x in l:
    if x == n:
      return True
  return False

def path_tail_cut(path: str) -> str:
  return ' '.join(path.split(' ')[0: -2])

def noun_in_chunk(c: Chunk) -> Morph:
  for m in c.morphs:
    if m.pos == '名詞':
      return m

def q49():
  # 1. for each noun_pair x,y in sentence:
  # 2. if: x is connected with y, output path from x to y
  # 3. else: for each ancestor k of x, if y have the same ancestor k, then
  # 4. output i -> k | j -> k | k
  for root in q44():
    for x,y in pairs(sorted(filter(lambda x: chunk_include_noun(x.chunk), all_nodes(root)), key=lambda x: x.chunk.id_in_sentence)):
      noun_in_chunk(x.chunk).surface = 'X'
      noun_in_chunk(y.chunk).surface = 'Y'
      # print(x.chunk.text() + ', ' + y.chunk.text())
      if(is_connected(x, y)):
        print(path_node_to_node(x, y))
      else:
        y_ancestors = node_ancestors(y)
        for k in node_ancestors(x):
          if k in y_ancestors:
            print(path_tail_cut(path_node_to_node(x, k)) + ' | ' + path_tail_cut(path_node_to_node(y, k)) + ' | ' + k.chunk.text())


q49()


# for root in q44():
#  print(dot(root))