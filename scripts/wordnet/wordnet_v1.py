import nltk
#nltk.download()

from nltk.corpus import wordnet
import numpy as np
import pandas as pd

def check_wup(tested_word, tested_against):
    w1 = wordnet.synsets(tested_word)
    w2 = wordnet.synset(tested_against)
    wup = []
    for i in w1:
        wup.append(i.wup_similarity(w2))
    return wup


wup_list = check_wup("wheel", "car.n.01")
print(max(wup_list))