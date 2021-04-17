import sys, re, os.path, string
from collections import Counter, defaultdict
from itertools import count
from math import log
#%%
model_path = sys.argv[1]
test_path = sys.argv[2]
files = [os.path.join(root, file) for root, dir, files in os.walk(test_path) for file in files if not dir]

classes = ['positive', 'negative', 'truthful', 'deceptive']
stopwords = [i.replace('\n', '') for i in list(open('stopwords3.txt', 'r'))]
stopwords = [i for i in stopwords if i not in ['not']]

with open(model_path, 'r') as f:
    weights = eval(f.readline().replace('\n', ''))
    bias = eval(f.readline().replace('\n', ''))
    train_vocab = eval(f.readline().replace('\n', ''))

wts_posneg = dict(sorted(weights.items(), key=lambda x: x[1][0], reverse=True)[:800]+
                  sorted(weights.items(), key=lambda x: x[1][0])[:800])
wts_trudec = dict(sorted(weights.items(), key=lambda x: x[1][1], reverse=True)[:1000]+
                  sorted(weights.items(), key=lambda x: x[1][1])[:1000])
# %%
def classification(files, stopwords, weights, bias, train_vocab):
    classify = defaultdict(list)
    for file in files:
        # Classification
        text = open(file, 'r').readline().replace('\n','').strip().lower()
        text = text.translate(str.maketrans('', '', string.punctuation.replace('\'','')+string.digits)).strip()
        tokens = [i for i in text.split() if i not in stopwords]
        for clf in range(2):
            if clf == 0:
                activation = sum(wts_posneg[word][clf]*train_vocab[word] for word in tokens if word in wts_posneg) + bias[clf]
                classify[file].append('positive' if activation > 0 else 'negative')
            elif clf == 1:
                activation = sum(wts_trudec[word][clf]*train_vocab[word] for word in tokens if word in wts_trudec) + bias[clf]
                classify[file].append('truthful' if activation > 0 else 'deceptive')
    return classify
#%%
output = classification(files, stopwords, weights, bias, train_vocab)

with open('percepoutput.txt','w') as f:
    for file in output:
        f.write(f'{output[file][1]} {output[file][0]} {file}' + '\n')