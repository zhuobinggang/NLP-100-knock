import torch as t
import torch.nn as nn

k = 3
head = 1

decoder = nn.TransformerDecoderLayer(k, head, 4, 0)
vocabulary = ['^','A','B','C','$']
memory = t.randn(1,1,k)
emb_raw = nn.Embedding(len(vocabulary), k)
emb = {} 
for index, word in enumerate(vocabulary):
    emb[word] = emb_raw(t.tensor(index))[None,None,:]
sm = nn.Softmax(2)
l = nn.Linear(3,4)

# Start

def get_inp_tenstor(words):
    inp = [emb[word] for word in words]
    inp = t.cat(inp, 0)
    return inp

def get_output_words(inp_tensor):
    o = decoder(inp_tensor, memory)
    o = l(o) # (seq_len, batch_num, k)
    o = sm(o) # (seq_len, batch_num, vocabulary_len)
    words_index = [t.max(word[0], 0)[1] for word in o]
    words = [vocabulary[index] for index in words_index]
    return words

def run_by_times(times,start ='^'):
    inp = [start]
    for i in range(times):
        words = get_output_words(get_inp_tenstor(inp))
        print(''.join(words))
        inp += [words[-1]]





