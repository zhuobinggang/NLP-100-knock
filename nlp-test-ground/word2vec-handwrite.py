import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



# Input: medium_text.txt
input_file = 'medium_text.txt'
# input_file = 'test.txt'
high_freq_non_sense_words = ['at', 'onli', 'is', 'on', 'but', 'hi',  'for', 'that', 'be', 'in', 'wa', 'it', 'a', 'to', 'and', 'of', 'the']

rate = 0.3

# Read file
stems = None
def init_stems():
    global stems
    text = ''
    with open(input_file) as f:
        text = f.read()
    
    # Remove all puntuation and number
    text = re.sub('[^A-Za-z ]', ' ', text)
    text = re.sub(' +', ' ', text)
    
    # Only retain stem
    ps = nltk.stem.PorterStemmer()
    stems = list(map(lambda x: ps.stem(x) ,text.split(' ')))
    # Remove non-sense words
    stems = list(filter(lambda x: x not in high_freq_non_sense_words, stems))

# stem_infos : {uniq_stem: {time, onehot}}
stem_infos = {}
def init_stem_infos():
    global stem_infos
    for s in stems:
        if s in stem_infos:
            stem_infos[s]['time'] += 1  
        else:
            stem_infos[s] = {'time': 1, 'onehot': None}

def sorted_stem_time_pairs():
    r = []
    for key in stem_infos.keys():
        r.append((key, stem_infos[key]['time']))
    r = sorted(r, key=lambda x: x[1])
    return r


# generate hot vectors
def set_one_hot_vecs():
    global stem_infos
    uniq_words = stem_infos.keys()
    uniq_word_lens = len(uniq_words)
    for i,word in enumerate(uniq_words):
        onehot = np.zeros(uniq_word_lens)
        onehot[i] = 1
        stem_infos[word]['onehot'] = onehot

# create w1
hidden_units = 10
w1 = None
# create w2
w2 = None
def reset_w1_w2():
    global w1, w2
    uniq_words_len = len(stem_infos.keys())
    w1 = np.random.rand(hidden_units, uniq_words_len)
    # create w2
    w2 = np.random.rand(uniq_words_len, hidden_units)

# centre, target are one-hot vectors
def once(centre, target):
    global w1, w2
    h = np.dot(w1, centre)
    z = np.dot(w2, h)
    exp_z = np.exp(z)
    exp_sum = np.sum(exp_z)
    # # Loss
    # exp_target = np.sum(exp_z * target) # consider target only
    # loss = - np.log(exp_target / exp_sum)
    # print('Before loss: ' + str(loss))

    # Backprop
    # Set all y = 0 excepting target one
    # y = (exp_z * target) / exp_sum
    y = exp_z / exp_sum
    dz = y - target
    dw2 = np.outer(dz, h)
    dh = np.dot(w2.T, dz)
    dw1 = np.outer(dh, centre)
    w2 -= rate * dw2
    w1 -= rate * dw1

    # # Recalculate Loss and show
    # h = np.dot(w1, centre)
    # z = np.dot(w2, h)
    # exp_z = np.exp(z)
    # exp_sum = np.sum(exp_z)
    # # Loss
    # exp_target = np.sum(exp_z * target) # consider target only
    # loss = - np.log(exp_target / exp_sum)
    # print('After loss: ' + str(loss))

def once_by_stem(centre, target):
    a = stem_infos[centre]['onehot']
    b = stem_infos[target]['onehot']
    once(a, b)

# iterate all word pairs and get cost func by forward prop
def one_epoch():
    window = 2
    ## One Epoch
    for index, word in enumerate(stems):
        j = index - window
        while j <= index+window:
            if j == index or j < 0 or j >= len(stems):
                pass
            else:
                # use pair to train
                centre, target = (stem_infos[word]['onehot'], stem_infos[stems[j]]['onehot'])
                once(centre, target)
            j += 1


def find_nearest(word):
    center_onehot = stem_infos[word]['onehot']
    center_vector = np.dot(w1, center_onehot)
    minimum = 999999
    nearest = None
    for key in stem_infos.keys():
        if key == word:
            pass
        else:
            onehot = stem_infos[key]['onehot']
            vector = np.dot(w1, onehot)
            d = np.linalg.norm(center_vector-vector)
            if d < minimum:
                minimum = d
                nearest = key
    return nearest

def run():
    init_stems()
    init_stem_infos()
    set_one_hot_vecs()
    reset_w1_w2()

w1_2d = None

def plot():
    global w1_2d
    labels = list(stem_infos.keys())
    pca = PCA(n_components=2)
    w1_2d = pca.fit_transform(w1.T)
    x,y = (w1_2d[:, 0],w1_2d[:, 1])
    plt.scatter(x, y)
    for i in range(0,len(labels)):
        plt.annotate(labels[i], (x[i], y[i]))
    plt.show()


