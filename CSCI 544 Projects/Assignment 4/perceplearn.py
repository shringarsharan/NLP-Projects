import sys, re, os.path, string
from collections import Counter, defaultdict
from itertools import count
from math import log
import random, numpy as np
#%%

train_path = sys.argv[1]
#train_path = 'train_data'
classes = ['positive','negative','truthful','deceptive']
#categories = ['pos_neg', 'tru_dec']
files = [os.path.join(root, file) for root,dir,files in os.walk(train_path) for file in files if not dir]
stopwords = [i.replace('\n','') for i in list(open('stopwords3.txt','r'))]
#stopwords.remove('not')
stopwords = [i for i in stopwords if i not in ['not']]

train = []
train_vocab = Counter()

for file in files:
    text = open(file, 'r').readline().replace('\n','').strip().lower()
    text = text.translate(str.maketrans('', '', string.punctuation.replace('\'','')+string.digits)).strip()
    tokens = [i for i in text.split() if i not in stopwords]
    train_vocab.update(tokens)
    cls = []
    brkn = file.split('/')
    if 'positive' in brkn[-4]:
        cls.append(1)
    else:
        cls.append(-1)
    if 'truthful' in brkn[-3]:
        cls.append(1)
    else:
        cls.append(-1)
    train.append((tokens,cls))
#%%
def vanilla_perceptron(train, train_vocab, max_iter):
    van_wts = {i:[0,0] for i in train_vocab.keys()}
    van_bias = [0,0]
    random.seed(42)
    random.shuffle(train)
    for iteration in range(max_iter):
        for example, y in train:
            activation = [0, 0]
            for clf in range(2):
                activation[clf] = sum(van_wts[word][clf]*train_vocab[word] for word in example) + van_bias[clf]
                if y[clf] * activation[clf] <= 0:
                    for word in example:
                        van_wts[word][clf] += y[clf]*train_vocab[word]
                    van_bias[clf] += y[clf]
        #print(f'Iteration: {iteration}\n Vanilla_wts: {list(van_wts.items())[:2]}\n Vanilla bias: {van_bias}')
    return van_wts, van_bias
#%%
def avg_perceptron(train, train_vocab, max_iter):
    avg_wts, running_wts = [{i:[0,0] for i in train_vocab.keys()} for i in range(2)]
    avg_bias, running_bias = [[0,0] for i in range(2)]
    c = 1
    random.seed(42)
    random.shuffle(train)
    for iteration in range(max_iter):
        random.shuffle(train)
        for example, y in train:
            activation = [0,0]
            for clf in range(2):
                activation[clf] = sum(avg_wts[word][clf]*train_vocab[word] for word in example) + avg_bias[clf]
                if y[clf]*activation[clf] <= 0:
                    for word in example:
                        avg_wts[word][clf] += y[clf]*train_vocab[word]
                        running_wts[word][clf] += y[clf]*c*train_vocab[word]
                    avg_bias[clf] += y[clf]
                    running_bias[clf] += y[clf]*c
            c += 1
    avgd_wts = {word: [(avg_wts[word][clf] - (running_wts[word][clf]/c)) for clf in range(2)] for word in avg_wts}
    avgd_bias = [avg_bias[clf] - (running_bias[clf]/c) for clf in range(2)]
    return avgd_wts, avgd_bias
#%%
van_wts, van_bias = vanilla_perceptron(train, train_vocab, 30)
avgd_wts, avgd_bias = avg_perceptron(train, train_vocab, 30)

with open('vanillamodel.txt', 'w') as f:
    f.write(str(van_wts)+'\n')
    f.write(str(van_bias)+'\n')
    f.write(str(train_vocab))

with open('averagedmodel.txt', 'w') as f:
    f.write(str(avgd_wts)+'\n')
    f.write(str(avgd_bias)+'\n')
    f.write(str(train_vocab))
f.close()

#%%
print("Learning Done")