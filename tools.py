import re
from nltk.stem import PorterStemmer

ps = PorterStemmer()
words = []

def remove_punctuations(sent):
    return re.sub(r'[^\w\s]' , "" , sent)
def stemming(word_list):
    return [ps.stem(w) for w in word_list]

def all_words(dt):
    for d in dt["data"]:
        for q in d["query"]:
            clean_q = remove_punctuations(q)
            word_list = clean_q.lower().split()
            stemming_list = stemming(word_list)
            words.extend(stemming_list)
    return list(set(words))
