# From http://www.peterbloem.nl/blog/transformers

import torch
from torch import nn
import torch.nn.functional as F

class SelfAttentionMasked(nn.Module):
  def __init__(self, k, heads=8):
    super().__init__()
    self.k, self.heads = k, heads
    self.tokeys = nn.Linear(k, k*heads, bias=False)
    self.toqueries = nn.Linear(k, k*heads, bias=False)
    self.tovalues = nn.Linear(k, k*heads, bias=False)
    self.unifyheads = nn.Linear(heads * k, k)
  def forward(self, x):
    b, t, k = x.size()
    h = self.heads
    queries = self.toqueries(x).view(b, t, h, k)
    keys    = self.tokeys(x)   .view(b, t, h, k)
    values  = self.tovalues(x) .view(b, t, h, k)
    keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
    queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
    values = values.transpose(1, 2).contiguous().view(b * h, t, k) # (b*h, t, k)
    queries = queries / (k ** (1/4))
    keys    = keys / (k ** (1/4))
    dot = torch.bmm(queries, keys.transpose(1, 2)) # (b*h, t, t)
    indices = torch.triu_indices(t, t, offset=1)
    dot[:, indices[0], indices[1]] = float('-inf')
    dot = F.softmax(dot, dim=2)
    out = torch.bmm(dot, values).view(b, h, t, k)
    out = out.transpose(1, 2).contiguous().view(b, t, h * k)
    return self.unifyheads(out)


class SelfAttention(nn.Module):
  def __init__(self, k, heads=8):
    super().__init__()
    self.k, self.heads = k, heads
    self.tokeys = nn.Linear(k, k*heads, bias=False)
    self.toqueries = nn.Linear(k, k*heads, bias=False)
    self.tovalues = nn.Linear(k, k*heads, bias=False)
    self.unifyheads = nn.Linear(heads * k, k)
  def forward(self, x):
    b, t, k = x.size()
    h = self.heads
    queries = self.toqueries(x).view(b, t, h, k)
    keys    = self.tokeys(x)   .view(b, t, h, k)
    values  = self.tovalues(x) .view(b, t, h, k)
    keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
    queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
    values = values.transpose(1, 2).contiguous().view(b * h, t, k) # (b*h, t, k)
    queries = queries / (k ** (1/4))
    keys    = keys / (k ** (1/4))
    dot = torch.bmm(queries, keys.transpose(1, 2)) # (b*h, t, t)
    dot = F.softmax(dot, dim=2)
    out = torch.bmm(dot, values).view(b, h, t, k)
    out = out.transpose(1, 2).contiguous().view(b, t, h * k)
    return self.unifyheads(out)



class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
    
        self.attention = SelfAttention(k, heads=heads)
    
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
    
        self.ff = nn.Sequential(
          nn.Linear(k, 4 * k),
          nn.ReLU(),
          nn.Linear(4 * k, k))


    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
    
        fedforward = self.ff(x)
        return self.norm2(fedforward + x)


class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()

        self.num_tokens = num_tokens # Q: 是Vocabulary的大小？
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        # The sequence of transformer blocks that does all the 
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        # Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, num_classes)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing 
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the 
                 classes (where c is the nr. of classes).
        """
        # generate token embeddings
        tokens = self.token_emb(x)
        b, t, k = tokens.size()

        # generate position embeddings
        positions = torch.arange(t) # (t)
        # self.pos_emb(positions) : (t, k)
        # [None, :, :] : (1, t, k)
        # expand(b, t, k) : (b, t, k) , just copy
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k) 

        x = tokens + positions
        x = self.tblocks(x)

        # Average-pool over the t dimension and project to class 
        # probabilities
        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)



