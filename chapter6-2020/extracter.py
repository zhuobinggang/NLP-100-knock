import numpy as np
import math

# filename = 'cut.csv'
filename = 'newsCorpora.csv'
train_file_name = 'train.txt'
valid_flie_name = 'valid.txt'
test_file_name = 'test.txt'


lines = None

def init_alllines():
    global lines
    with open(filename) as f:
        lines = f.readlines()

def filter_by_publishers():
    global lines
    publishers_need_to_extract = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
    lines = list(filter(lambda line: line.split('\t')[3] in publishers_need_to_extract, lines))

def shuffle():
    global lines
    np.random.shuffle(lines)

def lines_format():
    global lines
    lines = list(map(lambda x: '{cate}\t{title}'.format(cate=x.split('\t')[4], title=x.split('\t')[1]), lines))
    
def write_files():
    length = len(lines)
    percent_80 = math.floor(length * 0.8)
    percent_10 = math.floor((length - percent_80) / 2)
    train = lines[0:percent_80]
    valid = lines[percent_80:percent_80+percent_10]
    test = lines[percent_80+percent_10:]
    with open(train_file_name, "w") as f:
        f.write('\n'.join(train))
    with open(valid_flie_name, "w") as f:
        f.write('\n'.join(valid))
    with open(test_file_name, "w") as f:
        f.write('\n'.join(test))


def run():
    init_alllines()
    filter_by_publishers()
    shuffle()
    lines_format()
    write_files()

run()





