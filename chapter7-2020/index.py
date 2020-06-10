from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import KMeans
import numpy as np

# For plotint dendrogram Start
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
# For plotint dendrogram End


model = None

def init_model():
    global model
    print('Loading trained model...')
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print('Model is ready.')

def q60():
    print(model.get_vector('United_States'))


def q61():
    print(model.similarity('United_States','U.S.'))

def q62():
    print(model.most_similar(positive=['United_States'],topn=10))

def q63():
    print(model.most_similar(['Spain','Athens'],['Madrid'],10))

def q64():
    print('Finding most similar word & accuricy for each line...')
    lines_to_write = []
    counter = 0
    with open('questions-words.txt') as f:
        for line in f:
            words = line.split()
            if len(words) < 4:
                pass
            else:
                a,b,c,d = words
                # b - a + c
                target, acc = model.most_similar([b,c],[a],1)[0]
                words.append(target)
                words.append(str(acc))
                counter += 1
                print(f'{counter}: {target}')
            lines_to_write.append(' '.join(words))
    print('Start Writing to questions-words.txt')
    with open('questions-words.txt', 'w') as target_file:
        target_file.write('\n'.join(lines_to_write))


def q65():
    pass

def preprocess_q66():
    # Append similaty to the tail
    with open('wordsim353/combined.csv') as f:
        lines = f.readlines()
        lines = lines[1:]
        lines = [line.strip().split(',') for line in lines]
        for line in lines:
            a,b,c = line
            simil = model.similarity(a,b)
            line.append(str(simil))
    with open('wordsim353.processed.csv', 'w') as f:
        content = '\n'.join(list(map(lambda words: ','.join(words), lines))) 
        f.write(content)


def append_d(rank):
    counter = 1
    for item in rank:
        index, weight = item
        item.append(counter)
        counter += 1
    prev_items = []
    for item in rank:
        accumulated_length = len(prev_items)
        if accumulated_length > 0 and prev_items[0][1] == item[1]:
            prev_items.append(item)
        elif accumulated_length > 0 and prev_items[0][1] != item[1]:
            # empty the items
            mean = sum(map(lambda x: x[2], prev_items)) / accumulated_length
            for prev_item in prev_items:
                prev_item[2] = mean
            prev_items = [item]
        elif accumulated_length == 0:
            prev_items = [item]
    return rank



def q66():
    # Get rank1
    raw_origin_num = []
    raw_similarity = []
    with open('wordsim353.processed.csv') as f:
        for index,line in enumerate(f.readlines()):
            _,_,a,b = line.strip().split(',')
            raw_origin_num.append([index, float(a)])
            raw_similarity.append([index, float(b)])
    # Get index,d pairs
    rank1 = list(sorted(raw_origin_num, key= lambda pair: pair[1]))
    rank2 = list(sorted(raw_similarity, key= lambda pair: pair[1]))
    
    # Append d to ranks
    rank1 = append_d(rank1)
    rank2 = append_d(rank2)

    # Resort by index
    rank1 = list(sorted(rank1, key=lambda x: x[0]))
    rank2 = list(sorted(rank2, key=lambda x: x[0]))

    # Calculate Spearman's rank correlation coefficient
    d_squares = []
    n = len(rank1)
    for index in range(n):
        d_squares.append((rank1[index][2] - rank2[index][2]) ** 2)
    return 1 - (6 * sum(d_squares)) / (n ** 3 - n)

def get_unique_contries():
    # preprocess: extract all contry vectors
    contries = set()
    with open('questions-words.txt') as f:
        recording = False
        for line in f:
            if not recording and line.startswith(': capital-world'):
                recording = True
            elif recording and line.startswith(':'):
                recording = False
                break
            elif recording:
                words = line.strip().split()
                contries.add(words[2])
                contries.add(words[3])
    return list(contries)



def q67():
    cs = get_unique_contries()
    vectors = [model.get_vector(c) for c in cs]
    kmeans = KMeans(n_clusters=5, n_init=3).fit(vectors)
    labels = kmeans.labels_
    # TODO: plot the cluster out
    contry_label_pairs = [(c, labels[index]) for index,c in enumerate(cs)]
    contry_label_pairs = list(sorted(contry_label_pairs, key=lambda x: x[1]))
    return contry_label_pairs


def plot_dendrogram(model, **kwargs):
    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])
    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def q68():
    cs = get_unique_contries()
    vectors = [model.get_vector(c) for c in cs]
    dendro = AgglomerativeClustering(n_clusters=5)
    dendro = dendro.fit(vectors)
    plt.title('Hierarchical Clustering Dendrogram')
    plot_dendrogram(dendro, labels=cs)
    plt.show()


