import re
# import nltk
# ps = nltk.stem.PorterStemmer()

def formated_line(line):
    cate, title = line.split('\t')
    title = re.sub('[^A-Za-z 0-9]', ' ', title)
    title = re.sub(' +', ' ', title)
    title_words = list((map(lambda x: ps.stem(x.lower()), title.split(' '))))
    title = ' '.join(title_words)
    return f'{cate}\t{title}'


# Read train file, stemerize, remove all special characters, lowercase, and write to new file
def format_input():
    # filename = 'head_train.txt'
    # filename = 'train.txt'
    # filename = 'test.txt'
    filename = 'valid.txt'
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            print(formated_line(line))

# format_input()

# for all words appeared in these files, format them and build a map
def all_uniq_words():
    all_words = []
    for filename in ['train.formatted.txt', 'valid.formatted.txt', 'test.formatted.txt']:
        with open(filename) as f:
            for line in f.readlines():
                cate, title = line.split('\t')
                for word in title.split(' '):
                    all_words.append(word)
    return set(all_words)

for word in all_uniq_words():
    print(word)



