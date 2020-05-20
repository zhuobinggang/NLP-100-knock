from nltk import stem

#Initialize an empty list
my_stemmed_words_list = []

#Start with a list of words.
word_list = [s.strip() for s in 'duct, tape, works, anywhere, magic, worshiped'.split(',')]
word_list.sort()

#Instantiate a stemmer object.
my_stemmer_object=stem.snowball.EnglishStemmer()

#Loop through the words in word_list
for word in word_list:
    my_stemmed_words_list.append(my_stemmer_object.stem(word))

print(my_stemmed_words_list)


## Traning set: T = [anywher, duct, magic, work]
## “Duct tape works anywhere. Duct tape is magic and should be worshiped.”
## duct tape work anywher duct tape magic worship
## 'anywher', 'duct', 'magic', 'tape', 'work', 'worship'


