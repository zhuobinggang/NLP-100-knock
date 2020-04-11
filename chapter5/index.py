from string import Template

# path = 'ee.txt'
# path = 'dd.cabocha'
path = 'neko.cut.cabocha'

class Chunk:
  def __init__(self, morphs, dst, srcs):
    self.morphs = morphs
    self.dst = dst
    self.srcs = srcs
  def __str__(self):
    t = Template('morphs: $morphs, dst: $dst, srcs: $srcs')
    morphs = ','.join(map(lambda x: x.base, self.morphs)) 
    srcs = ','.join(self.srcs)
    return t.substitute(morphs=morphs, dst=self.dst, srcs=srcs)
  def text(self):
    return ''.join(map(lambda x: x.surface, self.morphs))

  

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
  for line in sentence:
    if line.startswith('*'):
      if len(morphs) > 0:
        chunks.append(Chunk(morphs=morphs.copy(), dst=dst, srcs=[]))
      dst=int(line.split(' ')[2].split('D')[0])
      morphs.clear()
    else:
      morphs.append(morph(line))
  chunks.append(Chunk(morphs=morphs.copy(), dst=dst, srcs=[]))
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
  def __init__(self, chunk):
    self.children = []
    self.chunk = chunk
  def __str__(self):
    return self.chunk.text()
  def dot(self):
    result = ''
    for node in self.children:
      result += node.chunk.text() + ' -> ' + self.chunk.text() + '\n'
      result += node.dot()
    return result

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
      chunk_node_pairs.append((chunk, Node(chunk))) 
    for chunk, node in chunk_node_pairs:
      if chunk.dst != -1:
        chunk_node_pairs[chunk.dst][1].children.append(node)
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
        

q45()


# for root in q44():
#  print(dot(root))