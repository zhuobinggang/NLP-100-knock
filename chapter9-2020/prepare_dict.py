import numpy as np

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

def get_samples():
    res = []
    with open('train.processed.txt') as f:
        for line in f:
            cate, title = line.strip().split(' ')
            words = title.split(',')
            onehots = [[x] for x in ids2onehot(words2ids(words))]
            res.append((cate_to_number(cate), onehots))
    # word1 = np.zeros(2000)
    # word1[1000] = 1
    # word2 = np.zeros(2000)
    # word2[100] = 1
    # words = [[word1], [word2]] # (seq_size, batch_size, input_size)
    # label = 1
    # res.append((label, words))
    return res









