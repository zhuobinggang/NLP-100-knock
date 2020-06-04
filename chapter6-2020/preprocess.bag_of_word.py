for filename in ['train.formatted.txt', 'valid.formatted.txt', 'test.formatted.txt']:
    word_count_map = {}
    with open(filename) as f:
        for line in f.readlines():
            cate, title = line.split('\t')
            for word in title.strip().split(' '):
                if word in word_count_map :
                    word_count_map[word] += 1
                else:
                    word_count_map[word] = 1

sorted_keys = sorted(word_count_map.keys(), key = lambda key: word_count_map[key])
sorted_keys = list(reversed(sorted_keys))

def find_first_index_bigger_than_1():
    for i in range(len(sorted_keys)):
      if word_count_map[sorted_keys[i]] > 1:
        print(i)
        print(sorted_keys[i])
        break

def print_high_frequency_words():
    for word in sorted_keys[0: 2000]:
        print(f'{word},{word_count_map[word]}')


