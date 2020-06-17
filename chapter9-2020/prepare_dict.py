import numpy as np
from gensim.models.keyedvectors import KeyedVectors

word_index_map = None
word_cnt_pairs = None

def init_word_index_map():
    global word_index_map, word_cnt_pairs
    word_cnt_map = {}
    with open('train.processed.txt') as f:
        for line in f:
            cate, title = line.strip().split(' ')
            words = title.split(',')
            for w in words:
                if w not in word_cnt_map:
                    word_cnt_map[w] = 1
                else:
                    word_cnt_map[w] += 1
    # Build word_cnt_pairs for sorting
    word_cnt_pairs = [(word, word_cnt_map[word]) for word in word_cnt_map.keys()]
    word_cnt_pairs = list(reversed(sorted(word_cnt_pairs, key=lambda p: p[1])))
    # Assign index
    word_index_map = {}
    for index,(word, cnt) in enumerate(word_cnt_pairs):
        if cnt < 2:
            word_index_map[word] = 0
        else:
            word_index_map[word] = index + 1




def words2ids(words):
    if word_index_map is None:
        init_word_index_map()
    return [word_index_map[word] for word in words]


def ids2onehot(ids):
    res = []
    for index in ids:
        a = np.zeros(2000)
        if index > 0 and index <= 2000:
            a[index - 1] = 1
        res.append(a)
    return np.array(res)

def cate_to_number(cate):
    the_map = {'e':0, 'b':1, 't':2, 'm':3}
    return the_map[cate]

vec300 = None

def init_vec300():
    global vec300
    print('Loading trained vec300...')
    vec300 = KeyedVectors.load_word2vec_format('../chapter7-2020/GoogleNews-vectors-negative300.bin', binary=True)
    print('Vec300 is ready.')


def word2vec(word):
    # if word not in the whole model then return None
    try:
        vec = vec300.get_vector(word)
        return vec
    except KeyError as e:
        print(f'{word} not in vec300')
        return None


def words_to_vecs(words):
    if vec300 is None:
        init_vec300()
    res = [word2vec(word) for word in words]
    res = list(filter(lambda x: x is not None, res))
    return res

def get_samples_onehot():
    res = []
    with open('train.processed.txt') as f:
        for line in f:
            cate, title = line.strip().split(' ')
            words = title.split(',')
            onehots = [[x] for x in ids2onehot(words2ids(words))]
            res.append((cate_to_number(cate), onehots))
    return res

def get_samples():
    res = []
    with open('train.processed.txt') as f:
        for line in f:
            cate, title = line.strip().split(' ')
            words = title.split(',')
            # onehots = [[x] for x in ids2onehot(words2ids(words))]
            vecs = [[x] for x in words_to_vecs(words)]
            res.append((cate_to_number(cate), vecs))
    return res

def limit_length(vecs, new_len=15):
    if len(vecs) > new_len:
        return vecs[:15]
    else:
        for _ in range(new_len - len(vecs)):
            vecs.append(np.zeros(300))
    return vecs

def get_samples_CNN():
    res = []
    with open('train.processed.txt') as f:
        for line in f:
            cate, title = line.strip().split(' ')
            words = title.split(',')
            # onehots = [[x] for x in ids2onehot(words2ids(words))]
            vecs = words_to_vecs(words)
            vecs = limit_length(vecs, 15)
            vecs = np.array(vecs, dtype=np.float32)
            vecs = vecs.T # (300, 15)
            res.append((cate_to_number(cate), vecs))
    return res



