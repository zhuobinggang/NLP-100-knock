from gensim.models.keyedvectors import KeyedVectors

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



                

